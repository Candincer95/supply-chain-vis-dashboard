import pandas as pd
import unicodedata

# Load raw dataset with ISO-8859-1 encoding
df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding="ISO-8859-1")

# Select the 24 relevant columns for visualization and ML
selected_cols = [
    "order date (DateOrders)", "shipping date (DateOrders)",
    "Days for shipping (real)", "Days for shipment (scheduled)",
    "Delivery Status", "Late_delivery_risk", "Shipping Mode",
    "Market", "Order Region", "Order Country", "Order City",
    "Customer Country", "Customer Segment", "Latitude", "Longitude",
    "Department Name", "Category Name", "Product Name",
    "Type", "Order Status", "Order Item Quantity",
    "Sales", "Order Item Discount Rate", "Order Profit Per Order"
]
df = df[selected_cols].copy()

# Remove duplicates and missing values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Normalize accented characters to standard ASCII first
def normalize_ascii(text):
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("utf-8").strip()

for col in ["Order City", "Order Country", "Customer Country"]:
    df[col] = df[col].apply(normalize_ascii)

# Translate all Portuguese country names to standardized English names
country_fixes = {
    "Estados Unidos": "United States", "EE. UU.": "United States",
    "SudAfrica": "South Africa", "Sudafrica": "South Africa",
    "Republica Dominicana": "Dominican Republic", "Espana": "Spain",
    "Etiopia": "Ethiopia", "Tailandia": "Thailand",
    "Reino Unido": "United Kingdom", "Francia": "France",
    "Filipinas": "Philippines", "Italia": "Italy",
    "Alemania": "Germany", "Malasia": "Malaysia",
    "Brasil": "Brazil", "Afganistan": "Afghanistan",
    "Arabia Saudi": "Saudi Arabia", "Argelia": "Algeria",
    "Azerbaiyan": "Azerbaijan", "Banglades": "Bangladesh",
    "Barein": "Bahrain", "Belgica": "Belgium",
    "Belice": "Belize", "Bielorrusia": "Belarus",
    "Birmania": "Myanmar", "Myanmar (Birmania)": "Myanmar",
    "Bosnia y Herzegovina": "Bosnia and Herzegovina", "Botsuana": "Botswana",
    "Butan": "Bhutan", "Camboya": "Cambodia",
    "Camerun": "Cameroon", "Catar": "Qatar",
    "Chipre": "Cyprus", "Corea del Sur": "South Korea",
    "Costa de Marfil": "Ivory Coast", "Croacia": "Croatia",
    "Dinamarca": "Denmark", "Egipto": "Egypt",
    "Emiratos Arabes Unidos": "United Arab Emirates", "Eslovaquia": "Slovakia",
    "Eslovenia": "Slovenia", "Finlandia": "Finland",
    "Grecia": "Greece", "Guadalupe": "Guadeloupe",
    "Guayana Francesa": "French Guiana", "Hungria": "Hungary",
    "Irak": "Iraq", "Irlanda": "Ireland",
    "Islandia": "Iceland", "Japon": "Japan",
    "Jordania": "Jordan", "Kazajistan": "Kazakhstan",
    "Kenia": "Kenya", "Kirguistan": "Kyrgyzstan",
    "Letonia": "Latvia", "Libano": "Lebanon",
    "Libia": "Libya", "Lituania": "Lithuania",
    "Luxemburgo": "Luxembourg", "Macedonia": "North Macedonia",
    "Marruecos": "Morocco", "Martinica": "Martinique",
    "Moldavia": "Moldova", "Noruega": "Norway",
    "Nueva Zelanda": "New Zealand", "Paises Bajos": "Netherlands",
    "Papua Nueva Guinea": "Papua New Guinea", "Polonia": "Poland",
    "Republica Centroafricana": "Central African Republic",
    "Republica Checa": "Czech Republic",
    "Republica del Congo": "Republic of the Congo",
    "Republica Democratica del Congo": "Democratic Republic of the Congo",
    "Rumania": "Romania", "Rusia": "Russia",
    "Singapur": "Singapore", "Siria": "Syria",
    "Suazilandia": "Eswatini", "Sudan del Sur": "South Sudan",
    "Suecia": "Sweden", "Suiza": "Switzerland",
    "Surinam": "Suriname", "Tayikistan": "Tajikistan",
    "Trinidad y Tobago": "Trinidad and Tobago", "Tunez": "Tunisia",
    "Turquia": "Turkey", "Ucrania": "Ukraine",
    "Yibuti": "Djibouti", "Zimbabue": "Zimbabwe"
}

df["Order Country"] = df["Order Country"].replace(country_fixes)
df["Customer Country"] = df["Customer Country"].replace(country_fixes)

# Parse dates and engineer temporal & logistics features
df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
df["shipping_date"] = pd.to_datetime(df["shipping date (DateOrders)"], errors="coerce")
df.dropna(subset=["order_date", "shipping_date"], inplace=True)

df["Order Year"] = df["order_date"].dt.year
df["Order Month"] = df["order_date"].dt.to_period("M").astype(str)
df["Shipping Delay"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
df.drop(columns=["order date (DateOrders)", "shipping date (DateOrders)"], inplace=True)

# Filter valid rows and sample 20,000 records
df = df[(df["Sales"] > 0) & (df["Order Item Quantity"] > 0)]
df_clean = df.sample(n=20000, random_state=42)
df_clean.to_csv("cleaned_supply_chain.csv", index=False, encoding="utf-8")

print("Done! Cleaned shape:", df_clean.shape)
print("Unique Order Countries:", sorted(df_clean["Order Country"].unique()))