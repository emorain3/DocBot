from langchain_community.document_loaders import PyPDFLoader

# Load your PDF
loader = PyPDFLoader("pdf/Pregnancy Labs Condensed - 2024-25.pdf")

# Convert each page into a Document object
pages= []
for page in loader.lazy_load():
    pages.append(page)

print("** number of pages: ** ", len(pages), "\n===============\n")  # number of pages

print("** 'pages' metadata: ** \n", f"{pages[0].metadata}", "\n===============\n") # preview text
print("** page 1 UNFORMATTED: ** \n", pages[0], "\n===============\n")
print(pages[0].page_content) 
