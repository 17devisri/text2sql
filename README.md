# Deep Learning Text-to-SQL Model from Scratch in PyTorch

A complete, modular, and educational deep learning project implementing a **Sequence-to-Sequence (Seq2Seq) Transformer** from scratch in PyTorch to translate Natural Language Questions into executable SQL queries, conditioned on database schema DDLs.

Trained on the **[Gretel Synthetic Text-to-SQL](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)** dataset.

---

## 📌 Project Architecture

```
                                      +------------------------------------+
                                      | Natural Language Question + Schema |
                                      +-----------------+------------------+
                                                        |
                                                        v
                                             +--------------------+
                                             |  SQL Tokenizer     |
                                             +----------+---------+
                                                        |
                                                        v
                                             +--------------------+
                                             | Positional Embed   |
                                             +----------+---------+
                                                        |
                                                        v
                                             +--------------------+
                                             | Transformer Encoder|
                                             | (Multi-Head Self-A)|
                                             +----------+---------+
                                                        |  (Memory)
                                                        |
                 +-------------------+                  |
                 | Target SQL (<sos>)|                  |
                 +---------+---------+                  |
                           |                            |
                           v                            |
                 +-------------------+                  |
                 | Transformer Dec   |<-----------------+
                 | (Masked Self-A +  |
                 |  Cross-Attention) |
                 +---------+---------+
                           |
                           v
                 +-------------------+
                 | Output Projection |
                 | (Target SQL Vocab)|
                 +---------+---------+
                           |
                           v
                Generated SQL Query:
         "SELECT name, SUM(volume) FROM ..."
```

### Key Technical Components Built from Scratch:
1. **Schema-Aware Preprocessing**: Automatically parses the database context and isolates `CREATE TABLE ...` definitions, stripping bulk `INSERT` statements to preserve token budget.
2. **SQL-Aware Tokenizer**: A custom regex & vocabulary tokenizer handling SQL operators (`>=`, `!=`, `<>`), punctuation, literals, and keywords as discrete tokens.
3. **Transformer Seq2Seq Architecture**:
   - Sinusoidal Positional Encoding
   - Scaled Dot-Product Multi-Head Attention ($h=8, d_{model}=256$)
   - Causal Look-Ahead Masking in the decoder
   - Pre-LN (Layer Normalization) residual connections
   - Feed-Forward Networks with GELU activation
4. **Alternative Architecture (LSTM with Attention)**:
   - Bidirectional LSTM Encoder + Bahdanau Additive Attention Decoder for baseline comparison.
5. **Autoregressive Decoding Engine**:
   - Greedy Search Decoding
   - Beam Search Decoding with length normalization
   - In-memory SQLite syntax verification

---

## 📂 Project Structure

```
ag/
├── src/
│   ├── __init__.py
│   ├── tokenizer.py            # SQL-aware custom tokenizer & vocabulary
│   ├── dataset.py              # Schema extractor, Gretel dataset loader, collate fn
│   ├── models.py               # Transformer & LSTM Seq2Seq built from scratch
│   ├── train.py                # PyTorch training loop, teacher forcing, cosine LR
│   ├── inference.py            # Greedy & Beam search decoding, SQLite validator
│   └── evaluate.py             # Exact Match, syntax validity, and BLEU metrics
├── download_data.py            # Script to stream and cache a clean dataset subset
├── run_train.py                # Main script to train and evaluate the model
├── demo.py                     # Interactive CLI demo with pre-canned examples
├── experimentation_notebook.ipynb # Interactive Jupyter notebook for lab experiments
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Download Dataset Subset
Download and preprocess a sample from HuggingFace (e.g., 5,000 samples for swift local training):
```bash
python download_data.py --samples 5000
```
This saves clean `train.json` and `val.json` in `data/`.

### 2. Train the Model
Train the Transformer model:
```bash
python run_train.py --samples 5000 --epochs 6 --batch_size 32 --d_model 256
```
The script will:
- Build source and target vocabularies in `checkpoints/`.
- Train the model using teacher forcing and label smoothing.
- Save the best model checkpoint to `checkpoints/best_model.pt`.
- Automatically evaluate the model on the validation set.

### 3. Run the Interactive Demo
Run inference on custom questions and schemas:
```bash
python demo.py
```
You can type any natural language question and database schema to inspect the generated SQL and its syntax validity.

### 4. Evaluate an Existing Checkpoint
```bash
python run_train.py --eval_only
```

---

## 📊 Evaluation Metrics

The evaluation module measures:
- **Exact Match (EM)**: Normalized character and token matching with ground truth.
- **SQLite Syntax Validity**: Whether the generated query is valid SQL according to the SQLite engine.
- **BLEU / N-gram Overlap**: Token-level precision against reference queries.

---

## 🔬 Coursework / Viva Highlights
- **Why Pre-LN instead of Post-LN?** Pre-LN places LayerNorm inside the residual stream before the self-attention/FFN blocks, providing smoother gradient flow and stable convergence without needing warmup schedules.
- **Why Masked Self-Attention in the Decoder?** Ensures autoregressive causality — preventing the decoder from attending to future tokens during training with teacher forcing.
- **Why Schema Conditioning?** SQL queries are meaningless without database structure. Prefixing queries with `Question: ... | Schema: CREATE TABLE ...` allows cross-attention heads to bind natural language phrases (e.g. "salesperson") to specific table attributes.
