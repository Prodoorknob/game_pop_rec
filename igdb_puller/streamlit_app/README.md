# Streamlit App Notes

This directory contains the interactive Streamlit experience plus helper
utilities. Key entry points:

- `streamlit_app.py` – main UI entry point.
- `igdb_helpers.py` – convenience functions to talk to IGDB using the
  shared client logic from the package.
- `recommender.py` – hybrid recommendation engine with optional Merlin
  embeddings.

## Merlin Offline Training Workflow

To keep the Streamlit deployment light, the recommender falls back to a
cosine-similarity engine unless a Merlin model checkpoint is available.
You can train that checkpoint offline on a workstation that has the
NVIDIA Merlin stack installed and then drop the weights into
`streamlit_app/artifacts` for inference-time usage.

1. **Install dependencies (local workstation)**

   ```bash
   # Example conda environment with GPU-enabled PyTorch & Merlin
   conda create -n igdb-merlin python=3.10
   conda activate igdb-merlin
   pip install merlin-models==23.08 pandas pyarrow
   ```

   Ensure your machine has a CUDA-compatible GPU or install the CPU-only
   wheels provided by Merlin.

2. **Fetch training data**

   You can let the training script pull data directly from IGDB (it uses
   the same `load_games_for_analytics` helper as the app) or point it to
   a previously exported dataset (CSV/Parquet).

3. **Run the training helper**

   ```bash
   export TWITCH_CLIENT_ID=...
   export TWITCH_CLIENT_SECRET=...

   python streamlit_app/train_merlin_offline.py \
     --max-rows 25000 \
     --output streamlit_app/artifacts/merlin_games.pt
   ```

   - Use `--dataset path/to/games.parquet` to train from a local dump.
   - Add `--force` to ignore an existing checkpoint and retrain.

4. **Deploy the weights**

   Copy the generated `merlin_games.pt` into the Streamlit deployment.
   The app automatically looks for a checkpoint at
   `streamlit_app/artifacts/merlin_games.pt`. To change the location set
   the environment variable `MERLIN_RECOMMENDER_CHECKPOINT` to an
   absolute path before launching Streamlit.

5. **Run Streamlit**

   ```bash
   streamlit run streamlit_app/streamlit_app.py
   ```

   When the checkpoint is available and Merlin is installed, the
   recommendations will display the backend label
   “Merlin two-tower embeddings”. Otherwise the system falls back to the
   similarity engine and continues to work.

## CLI Recap

If you need to regenerate the analytics dataset, the package-level CLI
(`igdb-puller`) can dump the relevant endpoints (see the repository root
`README.md` for details).
