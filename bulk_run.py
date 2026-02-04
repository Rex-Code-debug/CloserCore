import pandas as pd
import os
import time
from bt_card import run_battle_card_generator # Importing your existing agent

# 1. Setup Folders
output_folder = "reports"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def run_bulk_job(input_csv="companies.csv"):
    # 2. Load Inputs
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print("❌ Error: companies.csv not found. Please create it first.")
        return

    summary_results = []
    failed_companies = []
    
    print(f"🚀 Starting Bulk Batch for {len(df)} companies...")
    
    # 3. The Loop
    for index, row in df.iterrows():
        company = row["Company"]
        print(f"\n[{index+1}/{len(df)}] Processing: {company}...")
        
        try:
            # CALL YOUR AGENT
            result = run_battle_card_generator(company)
            
            if result:
                # Save Markdown to Folder
                filename = f"{output_folder}/{company}_Battle_Card.md"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(result.get("final_report", "No report"))
                
                # Add to Summary List
                # We extract the 'One Liner' from the description state
                pricing_data = result.get("pricing_info", [{}])[0]
                starter_price = pricing_data.get("starter_plan", {}).get("price", "Unknown") if pricing_data else "Unknown"

                summary_results.append({
                    "Company": company,
                    "Website": result.get("home_url"),
                    "Description": result.get("description"),
                    "Starter_Price": starter_price,
                    "Report_File": filename
                })
                print(f"✅ Finished {company}")
            else:
                print(f"⚠️ Agent returned empty result for {company}")
                failed_companies.append(company)
                
        except Exception as e:
            print(f"❌ CRITICAL ERROR on {company}: {str(e)}")
            failed_companies.append(company)
            
        # 4. Respect Rate Limits (Sleep)
        time.sleep(2)

    # 5. Save Final CSV
    if summary_results:
        final_df = pd.DataFrame(summary_results)
        final_df.to_csv("final_delivery.csv", index=False)
        print("\n\n✨ Batch Complete! Check 'final_delivery.csv' and 'reports/' folder.")
    
    if failed_companies:
        print(f"⚠️ Failed Companies: {failed_companies}")

if __name__ == "__main__":
    # Create dummy CSV if it doesn't exist for testing
    if not os.path.exists("companies.csv"):
        with open("companies.csv", "w") as f:
            f.write("Company\nSlack\nodoo\nsalesforce")
            
    run_bulk_job()