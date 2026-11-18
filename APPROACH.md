# Approach

The pipeline is a local, config-driven Python program. Each source config maps input IDs and canonical business fields (supplier, amount, description, date, and optional PO/status/currency) to CSV columns. It combines source rows into one output row per record, and the category taxonomy is loaded from CSV at runtime.

For A, an exact normalized PO reference shared with the ERP is treated as a cross-system match; the ERP receipt is retained and the tool-export row is excluded as `DUPLICATE`. Repeated tool invoices are de-duplicated using invoice, supplier, PO, line text, and gross amount. For B, configured line identity fields identify repeated exports. Supplier IDs and normalized item identity fields are hashed into stable keys when golden keys are not available. Where A labels provide golden keys, those are retained for ERP rows.

Amounts are parsed as numeric values; gross A tool amounts are divided by one plus the VAT rate, defaulting to 15%. B amounts are converted to USD using line currency rate (currency units per USD), then the monthly invoice-date table when the line rate is absent. If that currency/month is absent from the FX table, the posting month is the explicit fallback; a rate still unavailable stops the run. Configured statuses and non-procurement rules control exclusions.

Classification uses character n-gram TF-IDF and nearest-neighbour paths from available labels. Known labelled rows retain their supplied labels. When there are no labels, it compares the line text directly to the taxonomy path names. The tree depth and category names are never embedded in code. Confidence is the nearest cosine similarity.

On A's 37,029 labelled ERP rows, a deterministic 80/20 split (29,623 train, 7,406 test; random seed 42, stratified by leaf category) achieved 86.11% exact category-path accuracy using labelled nearest neighbours. The cold-start tree-only run achieved 13.29% exact path/leaf accuracy on those same labelled descriptions. This confirms it runs without labels, but also shows the substantial accuracy loss when only category names are available.

On the current machine, a combined A/B run took about 37 seconds for 106,577 records, comfortably under the 60-minute limit. These measurements cover the current supplied files and local environment. The implementation makes no LLM calls; API cost is $0.

The supplied kit has no A2, B2, or C inputs, so blind outputs and blind-test accuracy cannot be reported yet. The supplied A/B output files represent the provided build data only.
