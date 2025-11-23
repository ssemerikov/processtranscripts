import csv
import json
import sys

def csv_to_json(csv_file_path, json_file_path=None):
    """
    Convert a CSV file to JSON format.
    
    Args:
        csv_file_path: Path to the input CSV file
        json_file_path: Path to the output JSON file (optional)
    
    Returns:
        List of dictionaries representing the JSON data
    """
    json_data = []
    
    try:
        # Read the CSV file
        with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
            # Use DictReader to automatically use first row as keys
            csv_reader = csv.DictReader(csv_file)
            
            # Convert each row to a dictionary and add to the list
            for row in csv_reader:
                json_data.append(row)
        
        # If output file path is provided, write to file
        if json_file_path:
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(json_data, json_file, indent=2, ensure_ascii=False)
            print(f"Successfully converted {csv_file_path} to {json_file_path}")
        
        return json_data
    
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 2:
        print("Usage: python csv_to_json.py <input_csv_file> [output_json_file]")
        print("\nExample:")
        print("  python csv_to_json.py data.csv")
        print("  python csv_to_json.py data.csv output.json")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    json_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Convert CSV to JSON
    json_data = csv_to_json(csv_file, json_file)
    
    # If no output file specified, print to console
    if not json_file:
        print(json.dumps(json_data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()