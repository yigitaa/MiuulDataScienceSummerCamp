COPY customers
FROM '...\OLIST\olist_customers_dataset.csv'
WITH (FORMAT CSV, HEADER);

COPY sellers
FROM '...\OLIST\olist_sellers_dataset.csv'
WITH (FORMAT CSV, HEADER);

COPY products
FROM '...\OLIST\olist_products_dataset.csv'
WITH (FORMAT CSV, HEADER);

COPY orders
FROM '...\OLIST\olist_orders_dataset.csv'
WITH (FORMAT CSV, HEADER);

COPY order_payments
FROM '...\OLIST\olist_order_payments_dataset.csv'
WITH (FORMAT CSV, HEADER);

COPY order_reviews
FROM '...\OLIST\olist_order_reviews_dataset.csv'
WITH (FORMAT CSV, HEADER);

COPY order_items
FROM '...\OLIST\olist_order_items_dataset.csv'
WITH (FORMAT CSV, HEADER);

COPY product_translation
FROM '...\OLIST\product_category_name_translation.csv'
WITH (FORMAT CSV, HEADER);

COPY geo_location
FROM '...\OLIST\olist_geolocation_dataset.csv'
WITH (FORMAT CSV, HEADER);
