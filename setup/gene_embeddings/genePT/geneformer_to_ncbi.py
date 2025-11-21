import requests
from bs4 import BeautifulSoup
import html2text
import mygene
import json
import pickle
from tqdm import tqdm
mg = mygene.MyGeneInfo()

parts_to_remove = [
    "##  Summary\n",
    "NEW",
    'Try the newGene table',
    'Try the newTranscript table',
    '**',
    "\nGo to the top of the page Help\n"
]

def rough_text_from_gene_name(gene_number):
    
    # get url
    url = f"https://www.ncbi.nlm.nih.gov/gene/{gene_number}"
    # Send a GET request to the URL
    summary_text = ''
    soup = None
    try:
        response = requests.get(url, timeout=30)
    except requests.exceptions.Timeout:
        print('time out')
        return((summary_text,soup))
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the "summary" tab content by inspecting the page's structure
        summary_tab = soup.find('div', {'class': 'rprt-section gene-summary'})

        # Check if the "summary" tab content is found
        if summary_tab:
            # Convert the HTML to plain text using html2text
            html_to_text = html2text.HTML2Text()
            html_to_text.ignore_links = True  # Ignore hyperlinks

            # Extract the plain text from the "summary" tab
            summary_text = html_to_text.handle(str(summary_tab))
            # Remove the specified parts from the original text
            for part in parts_to_remove:
                summary_text = summary_text.replace(part, ' ')
                # Replace '\n' with a space
            summary_text = summary_text.replace('\n', ' ')

            # Reduce multiple spaces into one space
            summary_text = ' '.join(summary_text.split())
            # Print or save the extracted text
        else:
            print("Summary tab not found on the page.")
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
    return((summary_text,soup))

with open(f"/work/magroup/kaileyhu/Geneformer/geneformer/token_dictionary_gc95M.pkl", 'rb') as handle:
    token_dictionary = pickle.load(handle)

# example query to convert gene IDs into page ids for NCBI 
# vocab_gene_list_results = mg.querymany(sorted(vocab_gene_list), scopes='symbol', species='human')
token_dictionary_results = mg.querymany(sorted(token_dictionary.keys()), fields="symbol")


gene_name_to_summary_page = {}
for i, token in tqdm(enumerate(token_dictionary_results)):
    gene_name = token['query']
    if 'notfound' in token:
        continue
    page_id = token['_id']
    try:
        if gene_name not in gene_name_to_summary_page:
            parsed_text, unparsed_html = rough_text_from_gene_name(page_id)
            gene_name_to_summary_page[gene_name] = parsed_text
    except:
        time.sleep(60)
        if gene_name not in gene_name_to_summary_page:
            parsed_text, unparsed_html = rough_text_from_gene_name(page_id)
            gene_name_to_summary_page[gene_name] = parsed_text

    if i % 10000 == 0:
        with open (f"/work/magroup/kaileyhu/data/genePT/ncbi_gene_desc_at_{token}.pkl", "wb") as handle:
            print(f"\n\n at token {i}, saving results\n\n")
            pickle.dump(gene_name_to_summary_page, handle)
            gene_name_to_summary_page = {}
