````markdown
# 🎯 Precision Hybrid Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![AI Models](https://img.shields.io/badge/Models-E5%20%7C%20MPNet%20%7C%20CrossEncoder-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**An enterprise-grade AI tool for classifying e-commerce products into specific category paths with high precision.**

This application uses a **"Hybrid" architecture**, combining State-of-the-Art Deep Learning (Vector Search) with Human Logic (Guardrails & Keywords) to solve complex classification problems like distinguishing between *Apparel*, *Toys*, and *Home Goods* even when product titles are ambiguous.

---

## 🌟 Key Features

* **🧠 Advanced AI Brain:** Utilizes **E5-Base** and **MPNet** for semantic retrieval, plus a **Cross-Encoder** reranker to judge the best match with high accuracy.
* **🛡️ Smart Guardrails:**
    * **Universal Matcher:** Dynamic word overlap detection (e.g., "Soap" in title matches "Soap" category automatically).
    * **Clothing Logic:** Ensures T-shirts/Hoodies go to Apparel, never Toys or Farm categories.
    * **Ambiguity Checks:** Prevents false positives (e.g., ensures "Tulipan" does not trigger "Pan" kitchen rules).
* **📂 Universal Batch Processing:**
    * Supports **Excel (.xlsx)** and **CSV** files.
    * **Auto-detects** columns (`product_name`, `description`).
    * **Data Preservation:** Keeps all your original columns (`sku`, `price`, etc.) and appends predictions to the right.
* **⚡ High Performance:** Caches AI embeddings locally for instant subsequent runs.

---

## 🛠️ Installation

### 1. Prerequisites
* Python 3.8 or higher.
* (Optional) NVIDIA GPU (The script automatically switches to CPU if no GPU is found).

### 2. Clone the Repository
```bash
git clone [https://github.com/yourusername/hybrid-classifier.git](https://github.com/yourusername/hybrid-classifier.git)
cd hybrid-classifier
````

### 3\. Setup Project Structure

Ensure your folder looks like this:

```text
/hybrid-classifier
  ├── gradio_app.py          # Main application script
  ├── requirements.txt       # Dependencies
  └── data/                  # YOUR DATA FOLDER
      ├── categories.csv     # Your category tree (ID, Path)
      └── tags.json          # Keyword mapping file
```

### 4\. Install Dependencies

```bash
pip install -r requirements.txt
```

-----

## 🚀 Usage

### Running the App

```bash
python gradio_app.py
```

  * Wait for the message: `✅ System Ready.`
  * Open the link provided in your terminal (usually `http://127.0.0.1:7860`).

### 🔹 Mode 1: Single Prediction

Perfect for testing individual items or debugging logic.

1.  Type a **Product Title** (e.g., *"Little Lambs Unisex Hoodie"*).
2.  (Optional) Type a **Description**.
3.  Click **Classify**.
4.  View the **Winner**, **Confidence Score**, and the **Logic** used (e.g., `🏆 Rule: Clothing Guardrail`).

### 🔹 Mode 2: Batch Prediction

Perfect for processing thousands of products at once.

1.  **Upload:** Drag & Drop your `Products.xlsx` or `.csv` file.
2.  **Map Columns:** Select which column contains the **Title** and **Description** from the dropdowns.
3.  **Set Limit:** Choose how many results you want per product (Top 1, 5, 10, etc.).
4.  **Run:** Click **🚀 Process Batch**.
5.  **Download:** Once the progress bar finishes, download the `batch_results.csv` file.

-----

## 🧠 How It Works (The Pipeline)

When a product is analyzed, it goes through a **5-Stage Decision Pipeline**:

1.  **The Scouts (Retrieval):**

      * The system converts your product text into mathematical vectors.
      * It scans your 34,000+ categories to find the **Top 30** most similar candidates.

2.  **The Tag Engine:**

      * It checks `tags.json` for exact keyword matches (e.g., "Dishwasher" -\> "Appliances").
      * Matches receive a **+30 Point Boost**.

3.  **The Guardrails (Safety Net):**

      * **Universal Matcher:** Checks if words in the title exist in the category path. (Matches "Hammer" to "Tools/Hammers"). **+40 Points.**
      * **Context Rules:**
          * If Title has "Tee/Shirt" -\> Boost **Apparel**, Punish **Toys**.
          * If Title has "Salt & Pepper" -\> Boost **Kitchen**.

4.  **The Judge (Reranking):**

      * The **Cross-Encoder** model reads the Top 15 candidates like a human would.
      * It assigns a final "Confidence Score" (e.g., `99.99` for perfect matches, `-8.5` for bad ones).

5.  **Final Decision:**

      * The system picks the highest-scoring category and returns it as the **Winner**.

-----

## 📂 Customizing for Your Store

To use this with your own inventory, simply replace the files in the `data/` folder:

1.  **`categories.csv`**: Must have at least two columns: `Category ID` and `Category Path`.
2.  **`tags.json`**: A dictionary mapping IDs to keywords. You can generate this automatically using a script or create it manually.

-----

## ☁️ Deployment (Render/Hugging Face)

This app is ready for cloud deployment.

  * **Runtime:** Python 3.10+
  * **Command:** `python gradio_app.py`
  * **Hardware:** Recommended 2GB+ RAM (Starter Plan on Render) due to AI model size.

-----

## 🤝 Contributing

Contributions, issues, and feature requests are welcome\! Feel free to check the [issues page](https://www.google.com/search?q=https://github.com/yourusername/hybrid-classifier/issues).

## 📝 License

This project is [MIT](https://choosealicense.com/licenses/mit/) licensed.

```
```