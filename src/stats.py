
import unicodedata

import pandas as pd
from fuzzywuzzy import fuzz, process
from statsmodels.stats.outliers_influence import variance_inflation_factor


def clean_text(text):
    """Chuẩn hóa chuỗi: chữ thường, loại bỏ dấu, ký tự đặc biệt và khoảng trắng dư thừa."""
    if not isinstance(text, str):
        return ''
    text = text.lower().strip()
    text = ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')
    text = text.replace('-', ' ').replace('_', ' ').replace('.', '')
    return text



# ======== 🔍 2. Hàm so khớp (Fuzzy Matching) ========
def fuzzy_match(query, choices, scorer=fuzz.token_set_ratio, cutoff=80, top_n=1):
    """So khớp fuzzy cho một chuỗi query với danh sách lựa chọn."""
    matches = process.extract(query, choices, scorer=scorer, limit=top_n)
    return [(match, score) for match, score in matches if score >= cutoff]

# ======== 🎛️ 3. Multi-pass Matching (So khớp nhiều bước) ========
def multi_pass_match(query, choices):
    for cutoff in [90, 80, 70]:
        matches = fuzzy_match(query, choices, cutoff=cutoff, top_n=10)
        if matches:
            return matches[0]  # Trả về kết quả tốt nhất
    return None, 0

# ======== 🎚️ 4. Ánh xạ sản phẩm thông minh ========
def map_products(
    source_df, source_col, target_df, target_col,
    groupby=None, scorer=fuzz.token_set_ratio, cutoff=80, top_n=1
):
    """
    Ánh xạ sản phẩm thông minh giữa hai DataFrame.
    - Hỗ trợ groupby theo danh mục (nếu có).
    - Ánh xạ qua nhiều bước và tính điểm tương đồng.
    """
    results = []

    # Hàm ánh xạ trong từng nhóm danh mục
    def process_group(source_group, target_group):
        source_products = source_group[source_col].dropna().unique()
        target_products = target_group[target_col].dropna().unique()

        for source_product in source_products:
            clean_source = clean_text(source_product)
            best_match, score = multi_pass_match(clean_source, [clean_text(t) for t in target_products])

            if best_match:
                results.append({
                    'Category': source_group[groupby].iloc[0] if groupby else 'N/A',
                    source_col: source_product,
                    target_col: best_match,
                    'Score': score,
                    'Similarity %': f"{score}%"
                })

    # Nếu có groupby, groupby để xử lý từng danh mục
    if groupby and groupby in source_df.columns and groupby in target_df.columns:
        for category, source_group in source_df.groupby(groupby):
            target_group = target_df[target_df[groupby] == category]
            process_group(source_group, target_group)
    else:
        # Nếu không có danh mục, so khớp toàn bộ
        process_group(source_df, target_df)

    return pd.DataFrame(results)

def VIF(model):
    # Assuming X is the design matrix used to fit the model
    X = model.model.exog

    # Create a DataFrame to store VIF values
    vif_data = pd.DataFrame()
    vif_data['feature'] = model.model.exog_names

    # Calculate VIF for each feature
    vif_data['VIF'] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

    vif_data['R-squared'] = model.rsquared
    return vif_data