import gradio as gr
import pandas as pd
import numpy as np
import pickle
import json

# Load
with open('outputs/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('outputs/features.json', 'r') as f:
    FEATURES = json.load(f)

vendor_stats = pd.read_csv('outputs/vendor_features.csv')

# Encode vendor_tag_name
vendor_stats['vendor_tag_name'] = vendor_stats['vendor_tag_name'].astype('category').cat.codes

def recommend_restaurants(latitude, longitude, gender, language):
    rows = []
    for _, v in vendor_stats.iterrows():
        dist = np.sqrt(
            (latitude  - v['latitude'])**2 +
            (longitude - v['longitude'])**2
        )
        row = {
            'latitude_x'             : latitude,
            'longitude_x'            : longitude,
            'vendor_lat'             : v['latitude'],
            'vendor_lon'             : v['longitude'],
            'cust_vendor_distance'   : min(dist, 20),
            'gender'                 : 0 if gender == 'Male' else 1,
            'language'               : 0 if language == 'English' else 1,
            'vendor_tag_name'        : v['vendor_tag_name'],
            'vendor_total_orders'    : v.get('vendor_total_orders', 0),
            'vendor_avg_rating'      : v.get('vendor_avg_rating', 0),
            'vendor_avg_distance'    : v.get('vendor_avg_distance', 0),
            'vendor_avg_prep_time'   : v.get('vendor_avg_prep_time', 0),
            'vendor_avg_grand_total' : v.get('vendor_avg_grand_total', 0),
            'vendor_unique_customers': v.get('vendor_unique_customers', 0),
            'vendor_favorite_count'  : v.get('vendor_favorite_count', 0),
            'vendor_popularity_rank' : v.get('vendor_popularity_rank', 100),
            'vendor_repeat_ratio'    : v.get('vendor_repeat_ratio', 0),
            'vendor_id'              : v['vendor_id']
        }
        rows.append(row)

    input_df = pd.DataFrame(rows)
    probs = model.predict_proba(input_df[FEATURES])[:, 1]
    input_df['probability'] = probs

    top5 = input_df.nlargest(5, 'probability')[['vendor_id', 'probability']]
    top5['probability'] = (top5['probability'] * 100).round(2)
    top5.columns = ['Vendor ID', 'Match Score (%)']
    top5 = top5.reset_index(drop=True)
    top5.index += 1

    return top5

# UI
with gr.Blocks(title="Restaurant Recommender") as app:
    gr.Markdown("# 🍽️ Restaurant Recommendation Engine")
    gr.Markdown("Enter your location and preferences to get personalized restaurant recommendations!")

    with gr.Row():
        with gr.Column():
            lat  = gr.Slider(-2, 2, value=0.0, label="Latitude")
            lon  = gr.Slider(-2, 2, value=0.0, label="Longitude")
            gen  = gr.Dropdown(['Male', 'Female'], label="Gender")
            lang = gr.Dropdown(['English', 'Arabic'], label="Language")
            btn  = gr.Button("🔍 Get Recommendations", variant="primary")

        with gr.Column():
            output = gr.Dataframe(
                headers=['Vendor ID', 'Match Score (%)'],
                label="Top 5 Recommended Restaurants"
            )

    btn.click(
        fn=recommend_restaurants,
        inputs=[lat, lon, gen, lang],
        outputs=output
    )

app.launch(share=True)