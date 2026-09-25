import os
import sqlite3
import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pandas.testing import assert_frame_equal

# Project Setup

BASE_URL = "https://books.toscrape.com/"
ALL_BOOKS_URL = urljoin(BASE_URL, "catalogue/page-{}.html")

DATA_DIR = os.path.join("data_pipeline", "data")
OUTPUT_DIR = os.path.join("data_pipeline", "outputs")

RAW_CSV = os.path.join(DATA_DIR, "books_raw.csv")
CLEAN_CSV = os.path.join(DATA_DIR, "books_clean.csv")
DB_FILE = os.path.join(DATA_DIR, "zepto_books.db")

QUERY_OUTPUT = os.path.join(OUTPUT_DIR, "query_results.txt")
COMPARISON_OUTPUT = os.path.join(
    OUTPUT_DIR, "sql_join_vs_pandas_merge.csv"
)

README_FILE = os.path.join("data_pipeline", "README.md")

GBP_TO_INR = 105.50

MIN_BOOKS = 60
MIN_CATEGORIES = 3
NUMBER_OF_PAGES = 5

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


# data frame-1


def create_session():
    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/153.0 Safari/537.36"
        )
    })

    return session


def get_page(session, url, retries=3, timeout=20):
    """
    Download a webpage with retry and error handling.
    """

    last_error = None

    for attempt in range(1, retries + 1):

        try:
            response = session.get(
                url,
                timeout=timeout
            )

            response.raise_for_status()

            return response.text

        except requests.RequestException as error:

            last_error = error

            print(
                f"Request failed "
                f"(attempt {attempt}/{retries}): {url}"
            )

            if attempt < retries:
                time.sleep(1)

    raise RuntimeError(
        f"Unable to download page after {retries} attempts: "
        f"{url}\nError: {last_error}"
    )


# Data frame-2


def extract_rating(article):
    """
    Extract rating text such as One, Two, Three, Four, Five.
    """

    rating_element = article.find("p", class_="star-rating")

    if rating_element is None:
        return None

    classes = rating_element.get("class", [])

    for class_name in classes:

        if class_name in RATING_MAP:
            return class_name

    return None


def extract_price(article):
    """
    Keep original price text for RAW dataset.
    """

    price_element = article.find(
        "p",
        class_="price_color"
    )

    if price_element is None:
        return None

    return price_element.get_text(strip=True)


def extract_availability(article):
    """
    Keep original availability text for RAW dataset.
    """

    availability_element = article.find(
        "p",
        class_="instock availability"
    )

    if availability_element is None:
        return None

    return availability_element.get_text(
        " ",
        strip=True
    )


def scrape_books_from_page(
    session,
    page_url,
    category_name,
    books
):
    """
    Scrape one category/page.
    """

    html = get_page(session, page_url)

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    articles = soup.find_all(
        "article",
        class_="product_pod"
    )

    for article in articles:

        try:

            title_element = article.find("h3")

            if title_element is None:
                print("Warning: title not found. Skipping row.")
                continue

            link_element = title_element.find("a")

            if link_element is None:
                print("Warning: book link not found. Skipping row.")
                continue

            title = link_element.get(
                "title",
                ""
            ).strip()

            relative_url = link_element.get(
                "href",
                ""
            )

            source_url = urljoin(
                page_url,
                relative_url
            )

            price_raw = extract_price(article)

            rating_text = extract_rating(article)

            availability_raw = extract_availability(article)

            books.append({
                "title": title,
                "price_raw": price_raw,
                "rating_text": rating_text,
                "availability_raw": availability_raw,
                "category": category_name,
                "source_url": source_url
            })

        except Exception as error:

            print(
                f"Warning: Failed to parse a book: {error}"
            )


def scrape_books_catalog():
    """
    as per assaigment will Scrape first 5 pages of the All Products catalog.

    For each book, visit the book detail page to obtain
    the real category from the breadcrumb.

    This may gives approximately 100 books and real categories.
    """

    session = create_session()

    books = []

    print("=" * 60)
    print("STARTING BOOK SCRAPING")
    print("=" * 60)

    # STEP 1: Scrape first 5 pages of All Products 
    

    for page_number in range(
        1,
        NUMBER_OF_PAGES + 1
    ):

        page_url = ALL_BOOKS_URL.format(
            page_number
        )

        print(
            f"Scraping catalog page "
            f"{page_number}/{NUMBER_OF_PAGES}"
        )

        html = get_page(
            session,
            page_url
        )

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        articles = soup.find_all(
            "article",
            class_="product_pod"
        )

        for article in articles:

            try:

                title_element = article.find("h3")

                if title_element is None:
                    continue

                link_element = title_element.find("a")

                if link_element is None:
                    continue

                title = link_element.get(
                    "title",
                    ""
                ).strip()

                relative_url = link_element.get(
                    "href",
                    ""
                )

                source_url = urljoin(
                    page_url,
                    relative_url
                )

                price_raw = extract_price(
                    article
                )

                rating_text = extract_rating(
                    article
                )

                availability_raw = extract_availability(
                    article
                )

                books.append({
                    "title": title,
                    "price_raw": price_raw,
                    "rating_text": rating_text,
                    "availability_raw": availability_raw,
                    "category": None,
                    "source_url": source_url
                })

            except Exception as error:

                print(
                    f"Warning: Catalog parsing error: {error}"
                )

    print(
        f"Books collected from catalog pages: "
        f"{len(books)}"
    )

    # --------------------------------------------------------
    # STEP 2: Visit each book detail page to get REAL category
    # --------------------------------------------------------

    print("=" * 60)
    print("EXTRACTING REAL BOOK CATEGORIES")
    print("=" * 60)

    for index, book in enumerate(
        books,
        start=1
    ):

        try:

            html = get_page(
                session,
                book["source_url"]
            )

            soup = BeautifulSoup(
                html,
                "html.parser"
            )

            breadcrumb = soup.find(
                "ul",
                class_="breadcrumb"
            )

            category = None

            if breadcrumb:

                breadcrumb_links = breadcrumb.find_all("a")

                # Typical structure:
                # Home > Books > Category > Book
                if len(breadcrumb_links) >= 3:

                    category = (
                        breadcrumb_links[-1]
                        .get_text(strip=True)
                    )

            if not category:

                category = "Unknown"

            book["category"] = category

            print(
                f"[{index}/{len(books)}] "
                f"{book['title']} -> {category}"
            )

        except Exception as error:

            print(
                f"Warning: Could not get category "
                f"for {book['title']}: {error}"
            )

            book["category"] = "Unknown"

    df_raw = pd.DataFrame(books)

    # --------------------------------------------------------
    # Remove duplicate books
    # --------------------------------------------------------

    df_raw = df_raw.drop_duplicates(
        subset=["source_url"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if len(df_raw) < MIN_BOOKS:

        raise AssertionError(
            f"Only {len(df_raw)} books scraped. "
            f"Minimum required is {MIN_BOOKS}."
        )

    real_categories = (
        df_raw["category"]
        .replace("", pd.NA)
        .dropna()
        .nunique()
    )

    if real_categories < MIN_CATEGORIES:

        raise AssertionError(
            f"Only {real_categories} categories found. "
            f"Minimum required is {MIN_CATEGORIES}."
        )

    print("=" * 60)
    print("SCRAPING VALIDATION PASSED")
    print(f"Books: {len(df_raw)}")
    print(f"Categories: {real_categories}")
    print("=" * 60)

    return df_raw


# ============================================================
# CLEANING
# ============================================================

def clean_price(value):
    """
    Convert price such as £51.77 into float 51.77.
    """

    if pd.isna(value):
        return None

    try:

        text = str(value)

        text = (
            text
            .replace("£", "")
            .replace("Â", "")
            .strip()
        )

        return float(text)

    except (ValueError, TypeError):

        return None


def clean_rating(value):
    """
    Convert One/Five style rating to integer 1-5.
    """

    if pd.isna(value):
        return None

    text = str(value).strip()

    return RATING_MAP.get(
        text,
        None
    )


def clean_availability(value):
    """
    Convert availability text into boolean.
    """

    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    if "in stock" in text:
        return True

    if "out of stock" in text:
        return False

    return None


def clean_data(df_raw):
    """
    Create cleaned/enriched dataset.
    """

    df = df_raw.copy()

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------

    df["price_gbp"] = (
        df["price_raw"]
        .apply(clean_price)
    )

    # --------------------------------------------------------
    # Rating
    # --------------------------------------------------------

    df["rating"] = (
        df["rating_text"]
        .apply(clean_rating)
    )

    # --------------------------------------------------------
    # Availability
    # --------------------------------------------------------

    df["in_stock"] = (
        df["availability_raw"]
        .apply(clean_availability)
    )

    # --------------------------------------------------------
    # Missing numeric values
    # --------------------------------------------------------

    price_median = df["price_gbp"].median()

    if pd.isna(price_median):

        raise ValueError(
            "Unable to calculate price median."
        )

    df["price_gbp"] = (
        df["price_gbp"]
        .fillna(price_median)
    )

    rating_median = df["rating"].median()

    if pd.isna(rating_median):

        raise ValueError(
            "Unable to calculate rating median."
        )

    df["rating"] = (
        df["rating"]
        .fillna(
            int(round(rating_median))
        )
    )

    # --------------------------------------------------------
    # Missing availability
    #
    # Availability is categorical/boolean rather than numeric.
    # For a missing value, False is used conservatively.
    # --------------------------------------------------------

    df["in_stock"] = (
        df["in_stock"]
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------------
    # Title/category handling
    # --------------------------------------------------------

    df["title"] = (
        df["title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["category"] = (
        df["category"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    # Remove rows without title/category
    df = df[
        (df["title"] != "")
        &
        (df["category"] != "")
    ].copy()

    # --------------------------------------------------------
    # Rating validation
    # --------------------------------------------------------

    df["rating"] = (
        df["rating"]
        .round()
        .astype(int)
    )

    df.loc[
        ~df["rating"].between(1, 5),
        "rating"
    ] = int(round(rating_median))

    # --------------------------------------------------------
    # Fixed currency conversion
    # --------------------------------------------------------

    df["price_inr"] = (
        df["price_gbp"]
        * GBP_TO_INR
    ).round(2)
    # --------------------------------------------------------
    # Final column order
    # --------------------------------------------------------

    df_clean = df[
        [
            "title",
            "price_gbp",
            "rating",
            "in_stock",
            "category",
            "price_inr",
            "source_url"
        ]
    ].reset_index(drop=True)

    return df_clean


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_clean_dataset(df):
    """
    Validate all major dataset requirements.
    """

    required_columns = {
        "title",
        "price_gbp",
        "rating",
        "in_stock",
        "category",
        "price_inr",
        "source_url"
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    assert not missing_columns, (
        f"Missing columns: {missing_columns}"
    )

    assert len(df) >= MIN_BOOKS, (
        f"Only {len(df)} rows."
    )

    assert df["category"].nunique() >= MIN_CATEGORIES, (
        "Less than 3 categories."
    )

    assert pd.api.types.is_numeric_dtype(
        df["price_gbp"]
    )

    assert pd.api.types.is_numeric_dtype(
        df["price_inr"]
    )

    assert pd.api.types.is_integer_dtype(
        df["rating"]
    )

    assert df["rating"].between(
        1,
        5
    ).all()

    assert df["price_gbp"].notna().all()

    assert df["price_inr"].notna().all()

    assert df["title"].notna().all()

    expected_inr = (
        df["price_gbp"]
        * GBP_TO_INR
    ).round(2)

    assert (
        (df["price_inr"] - expected_inr)
        .abs()
        < 1e-9
    ).all()

    print("Clean dataset validation PASSED.")


# ============================================================
# DATABASE
# ============================================================

def create_database(df):
    """
    Create normalized SQLite database.
    """

    if os.path.exists(DB_FILE):

        os.remove(DB_FILE)

    conn = sqlite3.connect(
        DB_FILE
    )

    cursor = conn.cursor()

    cursor.execute(
        "PRAGMA foreign_keys = ON;"
    )

    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        );
        """
    )

    # --------------------------------------------------------
    # Books
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            source_url TEXT,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        );
        """
    )

    # --------------------------------------------------------
    # Insert categories
    # --------------------------------------------------------

    categories = (
        df["category"]
        .drop_duplicates()
        .tolist()
    )

    for category in categories:

        cursor.execute(
            """
            INSERT INTO categories(category_name)
            VALUES (?);
            """,
            (category,)
        )

    conn.commit()

    # --------------------------------------------------------
    # Create category mapping
    # --------------------------------------------------------

    category_df = pd.read_sql(
        """
        SELECT category_id, category_name
        FROM categories;
        """,
        conn
    )

    category_map = dict(
        zip(
            category_df["category_name"],
            category_df["category_id"]
        )
    )

    df_db = df.copy()

    df_db["category_id"] = (
        df_db["category"]
        .map(category_map)
    )

    assert df_db["category_id"].notna().all()

    # --------------------------------------------------------
    # Insert books
    # --------------------------------------------------------

    books_db = df_db[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category_id",
            "source_url"
        ]
    ].copy()

    books_db["in_stock"] = (
        books_db["in_stock"]
        .astype(int)
    )

    books_db.to_sql(
        "books",
        conn,
        if_exists="append",
        index=False
    )

    conn.commit()

    return conn


# ============================================================
# SQL QUERIES
# ============================================================

def run_sql_queries(conn):
    """
    Execute at least 5 SQL queries covering the required
    SQL concepts.
    """

    queries = {

        "Query_1_SELECT_WHERE_ORDER_BY_LIMIT":
        """
        SELECT title, price_inr
        FROM books
        WHERE price_inr > 1500
        ORDER BY price_inr DESC
        LIMIT 5;
        """,

        "Query_2_DISTINCT":
        """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating ASC;
        """,

        "Query_3_IN_BETWEEN":
        """
        SELECT title, rating, price_gbp
        FROM books
        WHERE rating IN (4, 5)
        AND price_gbp BETWEEN 10 AND 50
        ORDER BY price_gbp ASC
        LIMIT 5;
        """,

        "Query_4_AGGREGATION":
        """
        SELECT
            c.category_name,
            COUNT(b.book_id) AS total_books,
            ROUND(AVG(b.price_gbp), 2) AS average_price_gbp
        FROM categories c
        JOIN books b
            ON c.category_id = b.category_id
        GROUP BY c.category_id, c.category_name
        ORDER BY total_books DESC;
        """,

        "Query_5_PK_FK_JOIN":
        """
        SELECT
            b.title,
            b.price_inr,
            b.rating,
            c.category_name
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        ORDER BY
            b.rating DESC,
            b.price_inr ASC
        LIMIT 10;
        """
    }

    with open(
        QUERY_OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "====================================================\n"
        )
        file.write(
            "ZEPTO DATA PIPELINE - SQL QUERY RESULTS\n"
        )
        file.write(
            "====================================================\n\n"
        )

        for query_name, query_sql in queries.items():

            result = pd.read_sql(
                query_sql,
                conn
            )

            file.write(
                f"--- {query_name} ---\n"
            )

            file.write(
                "SQL:\n"
            )

            file.write(
                query_sql.strip()
            )

            file.write(
                "\n\nOUTPUT:\n"
            )

            file.write(
                result.to_string(
                    index=False
                )
            )

            file.write(
                "\n\n"
                + "=" * 60
                + "\n\n"
            )

    return queries


# ============================================================
# SQL JOIN VS PANDAS MERGE
# ============================================================

def compare_sql_and_pandas(conn, queries):
    """
    Reproduce SQL JOIN using pandas.merge and verify
    exact equivalence.
    """

    # --------------------------------------------------------
    # SQL JOIN
    # --------------------------------------------------------

    sql_join_df = pd.read_sql(
        queries["Query_5_PK_FK_JOIN"],
        conn
    )

    # --------------------------------------------------------
    # Read database tables using pd.read_sql
    # --------------------------------------------------------

    books_db_df = pd.read_sql(
        "SELECT * FROM books;",
        conn
    )

    categories_db_df = pd.read_sql(
        "SELECT * FROM categories;",
        conn
    )

    # --------------------------------------------------------
    # Reproduce JOIN using pandas.merge
    # --------------------------------------------------------

    pandas_merge_df = pd.merge(
        books_db_df,
        categories_db_df,
        on="category_id",
        how="inner"
    )

    pandas_merge_df = (
        pandas_merge_df
        .sort_values(
            by=["rating", "price_inr"],
            ascending=[False, True]
        )
        .head(10)
    )

    pandas_merge_df = pandas_merge_df[
        [
            "title",
            "price_inr",
            "rating",
            "category_name"
        ]
    ].reset_index(drop=True)

    sql_join_df = (
        sql_join_df
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Exact comparison
    # --------------------------------------------------------

    try:

        assert_frame_equal(
            sql_join_df,
            pandas_merge_df,
            check_dtype=True
        )

        match = True

    except AssertionError as error:

        match = False

        with open(
            QUERY_OUTPUT,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                "\nJOIN COMPARISON ERROR:\n"
            )

            file.write(
                str(error)
            )

        raise AssertionError(
            "SQL JOIN and pandas.merge results do not match."
        )

    # --------------------------------------------------------
    # Save side-by-side comparison
    # --------------------------------------------------------

    comparison_df = pd.concat(
        [
            sql_join_df.add_prefix(
                "SQL_"
            ),
            pandas_merge_df.add_prefix(
                "PANDAS_"
            )
        ],
        axis=1
    )

    comparison_df.to_csv(
        COMPARISON_OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # Save comparison evidence to text file
    # --------------------------------------------------------

    with open(
        QUERY_OUTPUT,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n====================================================\n"
        )

        file.write(
            "SQL pd.read_sql OUTPUT\n"
        )

        file.write(
            "====================================================\n"
        )

        file.write(
            sql_join_df.to_string(
                index=False
            )
        )

        file.write(
            "\n\n"
        )

        file.write(
            "====================================================\n"
        )

        file.write(
            "PANDAS pd.merge OUTPUT\n"
        )

        file.write(
            "====================================================\n"
        )

        file.write(
            pandas_merge_df.to_string(
                index=False
            )
        )

        file.write(
            "\n\n"
        )

        file.write(
            f"EXACT MATCH VERIFICATION: {match}\n"
        )

        file.write(
            "SQL JOIN and pandas.merge are equivalent.\n"
        )

    print(
        "Exact SQL JOIN vs pandas.merge match confirmed."
    )

    return match


# ============================================================
# DATABASE VALIDATION
# ============================================================

def validate_database(conn, expected_book_count):
    """
    Validate SQLite database and PK/FK relationship.
    """

    # --------------------------------------------------------
    # Table validation
    # --------------------------------------------------------

    tables = pd.read_sql(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        AND name IN ('books', 'categories')
        ORDER BY name;
        """,
        conn
    )

    table_names = set(
        tables["name"].tolist()
    )

    assert "books" in table_names
    assert "categories" in table_names

    # --------------------------------------------------------
    # Row counts
    # --------------------------------------------------------

    book_count = pd.read_sql(
        "SELECT COUNT(*) AS count FROM books;",
        conn
    ).iloc[0]["count"]

    category_count = pd.read_sql(
        "SELECT COUNT(*) AS count FROM categories;",
        conn
    ).iloc[0]["count"]

    assert book_count >= MIN_BOOKS
    assert book_count == expected_book_count
    assert category_count >= MIN_CATEGORIES

    # --------------------------------------------------------
    # Foreign key validation
    # --------------------------------------------------------

    invalid_fk = pd.read_sql(
        """
        SELECT COUNT(*) AS count
        FROM books b
        LEFT JOIN categories c
            ON b.category_id = c.category_id
        WHERE c.category_id IS NULL;
        """,
        conn
    ).iloc[0]["count"]

    assert invalid_fk == 0

    # --------------------------------------------------------
    # Rating validation
    # --------------------------------------------------------

    invalid_ratings = pd.read_sql(
        """
        SELECT COUNT(*) AS count
        FROM books
        WHERE rating < 1 OR rating > 5;
        """,
        conn
    ).iloc[0]["count"]

    assert invalid_ratings == 0

    # --------------------------------------------------------
    # NULL validation
    # --------------------------------------------------------

    null_values = pd.read_sql(
        """
        SELECT COUNT(*) AS count
        FROM books
        WHERE title IS NULL
           OR price_gbp IS NULL
           OR price_inr IS NULL
           OR rating IS NULL
           OR in_stock IS NULL
           OR category_id IS NULL;
        """,
        conn
    ).iloc[0]["count"]

    assert null_values == 0

    print("SQLite database validation PASSED.")

    return {
        "book_count": int(book_count),
        "category_count": int(category_count)
    }


# ============================================================
# README
# ============================================================

def generate_readme(
    book_count,
    category_count
):
    """
    Generate project README automatically.
    """

    readme_content = f"""# Zepto Data Pipeline Capstone

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

The pipeline scrapes the first {NUMBER_OF_PAGES} pages of the All Products catalog.

The resulting dataset contains at least {MIN_BOOKS} books.

Actual book categories are obtained from each book's detail-page breadcrumb.

---

## 3. Dataset

Books scraped: {book_count}

Categories: {category_count}

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
"""

    with open(
        README_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            readme_content
        )


# ============================================================
# FINAL VALIDATION
# ============================================================

def final_validation(
    df_clean,
    db_validation,
    comparison_match
):
    """
    Final project acceptance validation.
    """

    assert len(df_clean) >= MIN_BOOKS

    assert (
        df_clean["category"].nunique()
        >= MIN_CATEGORIES
    )

    assert (
        db_validation["book_count"]
        == len(df_clean)
    )

    assert (
        db_validation["category_count"]
        >= MIN_CATEGORIES
    )

    assert comparison_match is True

    assert os.path.exists(
        RAW_CSV
    )

    assert os.path.exists(
        CLEAN_CSV
    )

    assert os.path.exists(
        DB_FILE
    )

    assert os.path.exists(
        QUERY_OUTPUT
    )

    assert os.path.exists(
        COMPARISON_OUTPUT
    )

    assert os.path.exists(
        README_FILE
    )

    print()
    print("=" * 70)
    print("ALL AUTOMATED ACCEPTANCE CHECKS PASSED!")
    print("=" * 70)
    print(
        f"Books: {len(df_clean)}"
    )
    print(
        f"Categories: "
        f"{df_clean['category'].nunique()}"
    )
    print(
        f"Currency rate: "
        f"1 GBP = {GBP_TO_INR:.2f} INR"
    )
    print(
        "SQLite database: PASSED"
    )
    print(
        "SQL queries: PASSED"
    )
    print(
        "pd.read_sql: PASSED"
    )
    print(
        "pd.merge: PASSED"
    )
    print(
        "SQL vs pandas exact match: PASSED"
    )
    print(
        "README: PASSED"
    )
    print("=" * 70)
    print("FINAL STATUS: ALL CHECKS PASSED")
    print("=" * 70)


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():

    # --------------------------------------------------------
    # Create folders
    # --------------------------------------------------------

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print("ZEPTO DATA PIPELINE STARTED")
    print("=" * 70)

    # --------------------------------------------------------
    # Task 1: Scrape
    # --------------------------------------------------------

    df_raw = scrape_books_catalog()

    # Save TRUE RAW data
    try:
        df_raw.to_csv(RAW_CSV, index=False)
    except PermissionError:
        import time
        alt_path = RAW_CSV.replace(".csv", f"_{int(time.time())}.csv")
        print(f"Warning: {RAW_CSV} is locked. Writing raw data to {alt_path} instead.")
        df_raw.to_csv(alt_path, index=False)

    print(
        f"Raw dataset saved: {RAW_CSV}"
    )

    # --------------------------------------------------------
    # Task 2 + Task 3: Clean + Enrich
    # --------------------------------------------------------

    df_clean = clean_data(
        df_raw
    )

    validate_clean_dataset(
        df_clean
    )

    # Save clean dataset
    df_clean.to_csv(
        CLEAN_CSV,
        index=False
    )

    print(
        f"Clean dataset saved: {CLEAN_CSV}"
    )

    # --------------------------------------------------------
    # Task 4: SQLite
    # --------------------------------------------------------

    conn = create_database(
        df_clean
    )

    print(
        f"SQLite database created: {DB_FILE}"
    )

    try:

        # ----------------------------------------------------
        # Task 5: SQL
        # ----------------------------------------------------

        queries = run_sql_queries(
            conn
        )

        print(
            f"SQL results saved: {QUERY_OUTPUT}"
        )

        # ----------------------------------------------------
        # Task 6: pandas.read_sql + pandas.merge
        # ----------------------------------------------------

        comparison_match = (
            compare_sql_and_pandas(
                conn,
                queries
            )
        )

        print(
            f"Comparison saved: "
            f"{COMPARISON_OUTPUT}"
        )

        # ----------------------------------------------------
        # Database validation
        # ----------------------------------------------------

        db_validation = validate_database(
            conn,
            expected_book_count=len(df_clean)
        )

    finally:

        conn.close()

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------

    generate_readme(
        book_count=len(df_clean),
        category_count=df_clean[
            "category"
        ].nunique()
    )

    print(
        f"README generated: {README_FILE}"
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    final_validation(
        df_clean,
        db_validation,
        comparison_match
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_pipeline()

    
