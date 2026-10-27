# World Models

## Model-based RL
- High-level algorithm:
  1. Take a dataset $\mathcal{D}=\{ s,a,s' \}$ collected using base behavior policy $\pi_{\beta}(\cdot \mid s)$
  2. Learn a simulator $f(s,a)=s'$ (you ideally want stochastic simulator, but we'll go ahead with deterministic one for simplicity)
    Minimised using some objective. If deterministic, can be MSE and if probabilistic, can be log likelihood.
  3. Learn the policy inside the environment using any RL algorithm
    Can be model-free or planning and trajectory optimisation algorithms like MCTS
- Why might it not work?
  1. Collected data does not represent the whole distribution.
  2. World is difficult to simulate or learner is underparameterised/overly complex.
  3. Learned simulator might be computationally intensive and slow to simulate (eg. updating the simulated world at 3fps), Planning might be difficult in that environment (so even if we have a learned model of the world, it's not enough to execute control in that environment), Model representation might not be efficient (eg. pixel representation might not be the most efficient way to perform planning and control)
- How to modularise the problem?
  1. Statistics/Algorithms problem
  2. DL/Models problem (GPUs go brr...)
  3. Planning/Control/RL problem

## LeWM

### Implementation details
**Encoder**: maps a given frame $o_t$ into compact low-dimensional representation $z_t$
- $z_t=\text{enc}_{\theta}(o_t)$
- ViT-Tiny — patch size 14, 12 layers, 3 attention heads, hidden dim 192
- Observation embedding $z_t$ is constructed from the $\texttt{[CLS]}$ token embedding of last layer.
- Projection step maps $\texttt{[CLS]}$ token embedding into new represenation space using 1-layer MLP with *BatchNorm*. 

**Predictor**: autoregressively predicts the representation of next frame $\hat{z}_{t+1}$ given latent embedding $z_t$ and action $a_t$
- $\hat{z}_{t+1}=\text{pred}_{\phi}(z_t,a_t)$
- Action-conditioned which means predicting next representation requires action taken.
- Takes as input history of $N$ frames representations and predicts the next frame representation.
- ViT-Small — 6 layers, 16 causal attention heads, 10% dropout. Actions added through AdaLN at each layer. 
- Predictor follows a projector network with same implementations as encoder.

**Decoder**: Uses a transformer based decoder to visualize the latent prediction from the $\texttt{CLS}$ token embedding of the last encoder layer.
- $\texttt{CLS}$ embedding of dimension 192 is projected to decoder's hidden dimension, used as shared key, value in cross attention.
- Image of size $224\times 224$ with patch size $16$, corresponds to $(224/16)^{2}=196$ learnable query tokens, each representing a cell in $14\times 14$ grid.
- Each query token attends to projected $\texttt{CLS}$ embedding and outputs per-patch tokens.
- The resulting patch embeddings are linearly projected to $16\times 16 \times 3$ RGB pixels patches, and $196$ patches are reshaped into an image of size $224\times 224$.

### Notation

| Dim   | Meaning                                  | Example   |
| ----- | ---------------------------------------- | --------- |
| $B$   | Batch of environments / initial states   | 1         |
| $S$   | Candidate action plans per environment   | 300       |
| $H$   | Actual context frames supplied initially | 3         |
| $C$   | Image channels                           | 3 (RGB)   |
| $h,w$ | Image height and width                   | $224,224$ |
| $T$   | Future planning steps (action blocks)    | 5         |
| $A$   | Dimensions of one action block           | 10        |
| $D$   | Latent embedding dimension               | 192       |

### Objective 

$$\mathcal{L}_{\text{LeWM}}=\mathcal{L}_{\text{pred}}+\lambda\text{SIGReg}(Z)$$

- Teacher-forcing loss: $\mathcal{L}_{\text{pred}} = \|z_{t+1} - \hat{z}_{t+1}\|_2^2$
- Sketched-Isotropic Gaussian Regularisation: $\text{SIGReg}(Z) = \frac{1}{M}\sum_{m=1}^{M}T(h^{(m)})$

> [!NOTE] Why is SIGReg necessary?
> Without SIGReg, LeWM predicts constant embedding $\text{enc}_{\theta}(x)=c$ which minimises the loss to zero exactly, leading to collapse. To prevent this, they add an anti-collapse regularisation term that promotes feature diversity in the embedding space.

### SIGReg

SIGReg penalises the representation when it's trivial (concentrated, low-rank or low-variance) by encouraging the latent representation to match an isotropic Gaussian target distribution. Regulariser term is given by $\text{SIGReg}(Z)$, where $Z=\set{z_1,z_2,\dots,z_{N}}$ are batch of embeddings.

> [!WARNING] write more about sigreg.

### Planning

For planning, LeWM takes an initial observation $o_1$ and iteratively predict the latent states upto horizon $H$ with randomly initialized states. At training time, planning is optimised by minimising final latent prediction-matching objective,

$$\mathcal{L}_{\text{plan}}(\hat{z}_H)=\|z_{g}-\hat{z}_{H}\|_2^2,\quad z_g=\text{enc}_{\theta}(o_g), \hat{z}_{t+1}=\text{pred}_{\phi}(z_t,a_t)$$

where $z_g$ is the goal embedding, can be extracted using encoder from goal image. Rest of the terms are similar to those mentioned above. 

At inference time, planning correspond to finite-horizon optimal control problem, 

$$
a_{1:H}^{*}=\argmin_{a_{1:H}} \mathcal{L}_{\text{plan}}(\hat{z}_H)
$$

which is solved using CEM. Autoregressive rollout accumulates prediction errors (growing quadratically with Horizon), deteriorating quality of action sequence. LeWM extenuates this by using *Model-Predictive Control (MPC)*, executing only the first $K$ planned actions before replanning using next state-observation.

### CEM
Cross-Entropy Method (CEM) is a zeroth-order optimisation algorithm that iteratively samples a distribution over actions, evaluates the trajectories and updates the distribution to generate better actions.  

CEM starts with an initial distribution over actions (usually Gaussian $\mathcal{N}(0,I)$). In each iteration, it samples $n$ trajectories, evaluates them using the world model and selects the best $m$ trajectories. The parameters of the Gaussian distribution are then updated to match the mean and covariance of the selected trajectories. This process is repeated until the trajectories converge or a maximum number of iterations is reached. 

For each planning step, CEM samples 300 candidates, evaluates the actions using the learned model, and selects 30 best action sequences (known as *elites*). These action sequences are used to update the sampling distribution. Using MPC, LeWM keeps the horizon at 5 steps, which equates to 25 environment timesteps due to frame skips set to 5.

# Implementation
- LeJepa: https://arxiv.org/abs/2511.08544
  - https://rezabyt.github.io/blogposts/sigreg-tutorial.html
- LeWM: https://www.alphaxiv.org/abs/2603.19312
- Hierarchical planning JEPA: https://arxiv.org/abs/2604.03208
- SubJepa: https://arxiv.org/abs/2605.09241
- Fast-LeWM: https://arxiv.org/abs/2606.26217
- Temporal-distance JEPA: https://arxiv.org/abs/2607.25337
- PhyLatent: https://arxiv.org/abs/2608.05720
- FlowJepa: https://arxiv.org/abs/2608.29029
- LeFlow: https://arxiv.org/abs/2608.24855
- MIRA-WM: https://arxiv.org/abs/2607.05352
- LeWAM: https://www.alphaxiv.org/pdf/2610.12818
- https://github.com/galilai-group/stable-pretraining
- https://github.com/galilai-group/stable-worldmodel
- https://rezabyt.github.io/blogposts/sigreg-tutorial.html
- https://elonlit.com/scrivings/the-annotated-jepa/
- https://rohitbandaru.github.io/blog/JEPA-Deep-Dive/
- https://rohitbandaru.github.io/blog/World-Models/

# Usage

### 1. Local Training & Dataset Download

```bash
# Download dataset (e.g. tworooms, pusht, reacher, ogbench-cube)
uv run python src/dataset/download.py --dataset tworooms

# Train locally
uv run python src/train.py --run-id baseline --data-path "data/tworooms"
```

### 2. Environment Configuration (.env)

Copy [.env.example](file:///Users/apple/Projects/lonerapier/wm/.env.example) to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Key variables:
- `WANDB_API_KEY`: For Weights & Biases experiment tracking ([wandb.ai/authorize](https://wandb.ai/authorize))
- `HF_TOKEN`: Hugging Face write token for checkpoint storage ([huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))
- `HF_REPO_ID`: Target repository name (e.g., `username/wm-checkpoints`, created automatically if it doesn't exist)
- `VAST_API_KEY`: For provisioning and managing Vast.ai GPU instances ([cloud.vast.ai/account](https://cloud.vast.ai/account/))

### 3. Training on Vast.ai with W&B and Hugging Face

#### Option A: Automated from Local Machine (VastAI SDK)

Use [scripts/vast_manage.py](file:///Users/apple/Projects/lonerapier/wm/scripts/vast_manage.py) to search, launch, template, and sync to Vast.ai:

```bash
# Check account balance and authentication
uv run scripts/vast_manage.py user

# Search available RTX 4090 / A100 GPU offers under $0.50/hr
uv run scripts/vast_manage.py search --gpu RTX_4090 --max-dph 0.50

# Optional: List or create a reusable template (skips repeated environment setup)
uv run scripts/vast_manage.py templates
uv run scripts/vast_manage.py create-template --name "wm-trainer" --image "pytorch/pytorch:2.14.1-cuda12.6-cudnn9-devel" --disk 60

# Launch an instance (using base image or template)
uv run scripts/vast_manage.py launch <OFFER_ID> --disk 60
# Or with a template:
uv run scripts/vast_manage.py launch <OFFER_ID> --template <TEMPLATE_ID_OR_NAME>

# Check instance status and connection info
uv run scripts/vast_manage.py instances

# Sync local code and .env to the instance
uv run scripts/vast_manage.py sync <INSTANCE_ID>

# Connect via interactive SSH
uv run scripts/vast_manage.py ssh <INSTANCE_ID>

# Stop instance (pauses compute billing) or destroy when done
uv run scripts/vast_manage.py stop <INSTANCE_ID>
uv run scripts/vast_manage.py destroy <INSTANCE_ID>
```

#### Option B: On the Vast.ai Instance

Once connected to your Vast.ai instance:

```bash
# 1. Bootstrap environment, uv, CUDA, dependencies, and dataset (idempotent, <2s on warm runs):
bash scripts/setup_instance.sh tworooms

# 2. Run training with W&B logging & automatic Hugging Face checkpoint uploads:
bash scripts/train_vast.sh --run-id exp_01 --num-epochs 20 --batch-size 128

# Optional: Run in detached tmux session so you can disconnect:
bash scripts/train_vast.sh --tmux --run-id exp_01

# Optional: Auto-stop instance via VastAI SDK when training finishes to save money:
bash scripts/train_vast.sh --auto-stop --run-id exp_01
```

### 4. Managing Checkpoints with Hugging Face

Use [scripts/sync_hf.py](file:///Users/apple/Projects/lonerapier/wm/scripts/sync_hf.py) to manage remote checkpoints:

```bash
# Test Hugging Face connection and repository access
uv run scripts/sync_hf.py test

# List saved checkpoints across runs on Hugging Face
uv run scripts/sync_hf.py list

# Download the latest checkpoint for a run
uv run scripts/sync_hf.py download --run-id exp_01 --dest checkpoints/

# Upload a local file to Hugging Face
uv run scripts/sync_hf.py upload checkpoints/checkpoint_final.pt --run-id exp_01

# Resume training directly from Hugging Face
uv run python src/train.py --run-id exp_02 --resume-from checkpoint_final.pt
```

### 5. Instant Startup with Custom Docker Image (Optional)

To skip all bootstrap and installation steps across fresh instances, build and push [Dockerfile.vast](file:///Users/apple/Projects/lonerapier/wm/Dockerfile.vast):

```bash
# Build and push to your container registry
docker build -t your_user/wm:latest -f Dockerfile.vast .
docker push your_user/wm:latest

# Create template with your pre-baked image
uv run scripts/vast_manage.py create-template --name "wm-trainer" --image "your_user/wm:latest" --disk 60

# Any instance launched from this template boots in seconds with everything pre-installed!
uv run scripts/vast_manage.py launch <OFFER_ID> --template wm-trainer
```


# TODO
- recheck that num_frame = 4, horizon = 5, frameskip = 5, predictor history length = 3 for pusht, 1 for tworooms
- [x] bf16/fp16 training
- warmup + lambda decay scheduler
- eval the predicted plan, i.e. create the environment and perform planning, and then execute them.
  - executing the planned actions in the environment
- [x] Decoder to see latent states
- better planner, categorical CEM
- Try jepa variants — subjepa, flowjepa, fast-lewm, lewam.
- https://github.com/machines-in-motion/dreamer-v4
- does clipping to [-1, 1] for actions correct? what if action space is different?
- 