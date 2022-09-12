COPY customers
FROM 'X:\Miuul\pythonProject\OLIST\olist_customers_dataset.csv'
WITH (FORMAT CSV, HEADER);

COPY sellers
FROM 'X:\Miuul\pythonProject\OLIST\olist_sellers_dataset.csv'
WITH (FORMAT CSV, HEADER);

COPY products
FROM 'X:\Miuul\pythonProject\OLIST\olist_products_dataset.csv'
WITH (FORMAT CSV, HEADER);

COPY orders
FROM 'X:\Miuul\pythonProject\OLIST\olist_orders_dataset.csv'
WITH (FORMAT CSV, HEADER);

COPY order_payments
FROM 'X:\Miuul\pythonProject\OLIST\olist_order_payments_dataset.csv'
WITH (FORMAT CSV, HEADER);

COPY order_reviews
FROM 'X:\Miuul\pythonProject\OLIST\olist_order_reviews_dataset.csv'
WITH (FORMAT CSV, HEADER);

COPY order_items
FROM 'X:\Miuul\pythonProject\OLIST\olist_order_items_dataset.csv'
WITH (FORMAT CSV, HEADER);

COPY product_translation
FROM 'X:\Miuul\pythonProject\OLIST\product_category_name_translation.csv'
WITH (FORMAT CSV, HEADER);

COPY geo_location
FROM 'X:\Miuul\pythonProject\OLIST\olist_geolocation_dataset.csv'
WITH (FORMAT CSV, HEADER);