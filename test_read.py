import re

# Example JSON data
json_data = '''
{
  "id": "batch_req_XCKzWtdCfHi859CFLMHmGtPO",
  "custom_id": "|data|lhyman6|data|images|mss511850152-529.jpg",
  "response": {
    "status_code": 200,
    "request_id": "64d58ec80fb112faf8d0daff5d07054e",
    "body": {
      "id": "chatcmpl-9jSuI430GwH9u2aXz3nU9KWmbCXh5",
      "object": "chat.completion",
      "created": 1720622906,
      "model": "gpt-4o-2024-05-13",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "Here's the OCR extracted text in JSON format, followed by the summary and other required fields:\n\n```json\n{\n  \"ocr_text\": \"Miss Beatrice Vaughn,\\nc/o Toledo Blade,\\nToledo, Ohio.\\nDear Madam:\\n\\nYour favor of the 3rd instant is received together with\\nthe several clippings from the Toledo Blade of last November, and\\nfor which you will please accept my thanks. I was glad to hear from\\nyou and read your letter with interest.\\n\\nI am gratified with the sentiments you express in regard\\nto the labor movement, and your desire to be actively identified\\ntherewith.\\n\\nI should like to comply with your request for a commission\\nas general organizer, but one of the requirements for holding such\\na commission is that the applicant shall be a member in good standing\\nof the trade he or she represents, affiliated with the American Federa-\\ntion of Labor. I take it from your letter and also from the conver-\\nsation I had with you during the Toronto Convention that you do\\nnot hold membership in an organization affiliated with the American\\nFederation of Labor. However, I feel confident that with the oppor-\\ntunities constantly arising in your work as a newspaper representa-\\ntive, that you can not only manifest your sympathy with and loyalty\\ntoward the labor movement, but can be of real help, particularly\\nthrough the articles you write for your paper.\",\n  \"summary\": \"The author acknowledges receiving Miss Beatrice Vaughn's letter and newspaper clippings, expressing appreciation for her support of the labor movement. The author explains the requirement that a general organizer must be a member of the American Federation of Labor, which Miss Vaughn is not. However, the author encourages her to continue supporting the labor movement through her work as a newspaper representative.\",\n  \"date\": \"1910-01-11\",\n  \"is_printed\": true,\n  \"is_legible\": true\n}\n```"
          },
          "logprobs": null,
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 983,
        "completion_tokens": 409,
        "total_tokens": 1392
      },
      "system_fingerprint": "fp_4008e3b719"
    }
  },
  "error": null
}
'''

# Regex patterns to extract fields
ocr_text_pattern = r'"ocr_text": "([^"]+)"'
summary_pattern = r'"summary": "([^"]+)"'
date_pattern = r'"date": "([^"]+)"'
is_printed_pattern = r'"is_printed": (true|false)'
is_legible_pattern = r'"is_legible": (true|false)'

# Find matches using regex
ocr_text_match = re.search(ocr_text_pattern, json_data)
summary_match = re.search(summary_pattern, json_data)
date_match = re.search(date_pattern, json_data)
is_printed_match = re.search(is_printed_pattern, json_data)
is_legible_match = re.search(is_legible_pattern, json_data)

# Extract and print matched values
if ocr_text_match:
    ocr_text = ocr_text_match.group(1)
    print(f"OCR Text: {ocr_text}")

if summary_match:
    summary = summary_match.group(1)
    print(f"Summary: {summary}")

if date_match:
    date = date_match.group(1)
    print(f"Date: {date}")

if is_printed_match:
    is_printed = is_printed_match.group(1)
    print(f"Printed: {is_printed}")

if is_legible_match:
    is_legible = is_legible_match.group(1)
    print(f"Legible: {is_legible}")
