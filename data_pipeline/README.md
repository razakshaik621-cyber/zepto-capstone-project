# Zepto Data Pipeline Capstone

## 1. Project Overview

This project implements an end-to-end data pipeline using the public practice website:

https://books.toscrape.com/

The pipeline performs:

1. Web scraping
2. Raw data storage
3. Data cleaning
4. Data enrichment
5. GBP to INR conversion
6. SQLite normalization
7. SQL analysis
8. pandas verification
9. SQL JOIN vs pandas.merge equivalence testing

---

## 2. Data Source

Source:

https://books.toscrape.com/

No login, API key, or private data is required.

The pipeline scrapes the first 5 pages of the All Products catalog.

The resulting dataset contains at least 60 books.

Actual book categories are obtained from each book's detail-page breadcrumb.

---

## 3. Dataset

Books scraped: 100

Categories: 29

Required columns in cleaned dataset:

- title
- price_gbp
- rating
- in_stock
- category
- price_inr
- source_url

---

## 4. Cleaning Decisions

### Price

Example source value:

£51.77

Cleaning:

- Remove £ symbol
- Remove encoding artifact if present
- Convert to float

Final column:

price_gbp

### Rating

Source values:

One, Two, Three, Four, Five

Converted to:

1, 2, 3, 4, 5

### Availability

Source availability text is converted to:

True / False

Missing availability is treated as False.

### Missing Numeric Values

Missing price values are replaced using the median price.

Missing rating values are replaced using the median rating.

Rows without a valid title are removed.

These decisions prevent malformed numeric values from stopping the pipeline while preserving the majority of scraped records.

---

## 5. Currency Conversion

A fixed project baseline is used.

**1 GBP = 105.50 INR**

Formula:

price_inr = price_gbp * 105.50

No live currency API is used.

---

## 6. SQLite Database

Database:

data_pipeline/data/zepto_books.db

### categories table

- category_id - Primary Key
- category_name - Unique category name

### books table

- book_id - Primary Key
- title
- price_gbp
- price_inr
- rating
- in_stock
- category_id - Foreign Key
- source_url

Relationship:

books.category_id -> categories.category_id

---

## 7. SQL Queries

The pipeline executes five SQL queries.

### Query 1

SELECT + WHERE + ORDER BY + LIMIT

### Query 2

DISTINCT

### Query 3

IN + BETWEEN

### Query 4

JOIN + GROUP BY + COUNT + AVG

### Query 5

PK/FK JOIN

All SQL strings and outputs are saved in:

data_pipeline/outputs/query_results.txt

---

## 8. pandas Verification

The pipeline uses `pd.read_sql()` to read SQL results and database tables.

The SQL JOIN result is independently reproduced using:

pd.merge()

The SQL and pandas results are compared using:

assert_frame_equal()

The comparison result is saved in:

data_pipeline/outputs/query_results.txt

A side-by-side CSV is also generated:

data_pipeline/outputs/sql_join_vs_pandas_merge.csv

---

## 9. Generated Files

data_pipeline/

    data/

        books_raw.csv

        books_clean.csv

        zepto_books.db

    outputs/

        query_results.txt

        sql_join_vs_pandas_merge.csv

    scrape_pipeline.py

    README.md

---

## 10. Installation

Open PowerShell in the project root.

Create/activate virtual environment if required.

Install dependencies:

pip install pandas requests beautifulsoup4

---

## 11. Run Pipeline

From the project root:

python data_pipeline/scrape_pipeline.py

The complete pipeline runs automatically.

No manual copy/paste of scraped data is required.

---

## 12. Acceptance Checks

The pipeline automatically verifies:

- At least 60 books
- At least 3 categories
- Required cleaned columns
- Correct numeric types
- Ratings between 1 and 5
- Fixed GBP to INR conversion
- SQLite books table
- SQLite categories table
- Primary key / foreign key relationship
- SQL queries
- SQL JOIN
- pandas.merge
- Exact SQL/pandas equivalence

If a required check fails, the pipeline raises an error instead of reporting success.

---

## 13. Final Status

The pipeline prints:

ALL AUTOMATED ACCEPTANCE CHECKS PASSED

when all automated validations succeed.
