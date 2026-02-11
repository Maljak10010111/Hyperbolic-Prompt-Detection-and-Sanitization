import csv
from deep_translator import GoogleTranslator
import time

def translate_csv(input_file, output_file, target_language='es', columns_to_translate=None, batch_size=10, delay_between_batches=2):
    """
    Translate specific columns in a CSV file using batch translation
    
    Parameters:
    - input_file: Path to input CSV file
    - output_file: Path to output CSV file
    - target_language: Target language code (e.g., 'es' for Spanish, 'fr' for French)
    - columns_to_translate: List of column names to translate (None = translate all)
    - batch_size: Number of texts to translate per batch (default: 10, smaller = safer)
    - delay_between_batches: Seconds to wait between batches (default: 2)
    """
    
    translator = GoogleTranslator(source='auto', target=target_language)
    
    # Read the CSV file
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    # Determine which columns to translate
    if columns_to_translate is None:
        columns_to_translate = fieldnames
    
    # Initialize translated rows with original data
    translated_rows = [row.copy() for row in rows]
    
    # Process each column separately for batch translation
    for column in columns_to_translate:
        if column not in fieldnames:
            continue
            
        # Collect all texts from this column (including empty ones)
        texts_to_translate = []
        indices_with_text = []
        
        for idx, row in enumerate(rows):
            if column in row:
                text = row[column] if row[column] else ""
                texts_to_translate.append(text)
                indices_with_text.append(idx)
        
        if not texts_to_translate:
            continue
        
        # Filter out empty texts for actual translation
        non_empty_texts = []
        non_empty_indices = []
        for idx, text in enumerate(texts_to_translate):
            if text and text.strip():
                non_empty_texts.append(text)
                non_empty_indices.append(idx)
        
        if not non_empty_texts:
            continue
        
        print(f"\nTranslating column '{column}' ({len(non_empty_texts)} items)...")
        
        # Split into batches to respect rate limits
        total_batches = (len(non_empty_texts) + batch_size - 1) // batch_size
        
        for batch_num in range(0, len(non_empty_texts), batch_size):
            batch_texts = non_empty_texts[batch_num:batch_num + batch_size]
            batch_indices = non_empty_indices[batch_num:batch_num + batch_size]
            
            current_batch = (batch_num // batch_size) + 1
            print(f"  Processing batch {current_batch}/{total_batches} ({len(batch_texts)} items)...")
            
            try:
                # Batch translate using deep-translator
                translations = translator.translate_batch(batch_texts)
                
                # Map translations back to rows and display progress
                for text_idx, original_indices_idx in enumerate(batch_indices):
                    row_idx = indices_with_text[original_indices_idx]
                    original_text = batch_texts[text_idx]
                    translated_text = translations[text_idx]
                    translated_rows[row_idx][column] = translated_text
                    print(f"    ✓ {original_text[:50]}... → {translated_text[:50]}...")
                
                # Wait to respect rate limit
                if current_batch < total_batches:
                    print(f"  Waiting {delay_between_batches} seconds to respect rate limits...")
                    time.sleep(delay_between_batches)
                
            except Exception as e:
                print(f"  Error in batch {current_batch}: {e}")
                print(f"  Falling back to individual translation for this batch...")
                
                # Fallback to individual translation if batch fails
                for text_idx, original_indices_idx in enumerate(batch_indices):
                    try:
                        row_idx = indices_with_text[original_indices_idx]
                        original_text = batch_texts[text_idx]
                        translation = translator.translate(original_text)
                        translated_rows[row_idx][column] = translation
                        print(f"    ✓ {original_text[:50]}... → {translation[:50]}...")
                        time.sleep(1)
                    except Exception as e2:
                        print(f"    Error translating row {row_idx}: {e2}")
                        print(f"    Waiting 3 seconds before continuing...")
                        time.sleep(3)
    
    print(f"\n✓ Translation complete! Processed {len(translated_rows)} rows")
    
    # Write to output CSV - only translated content
    with open(output_file, 'w', encoding='utf-8', newline='') as file:
        print(f"✓ Saving output to: {output_file}...")
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(translated_rows)
    
    print(f"✓ Output saved to: {output_file}")


# Example usage:
if __name__ == "__main__":
    # Example 1: Translate all columns to French with batch size 10
    translate_csv('./visu_validation.csv',
                  './visu_validation_translated_it.csv',
                  target_language='it',
                  batch_size=10, 
                  delay_between_batches=1)