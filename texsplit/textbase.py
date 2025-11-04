# from langchain_text_splitters import RecursiveCharacterTextSplitter

# text = """
# Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.
# These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth.
# """

# splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
# chunks = splitter.split_text(text)

# print(len(chunks))
# print(chunks)


from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"C:\FILES\projects\langchain chatbot\texsplit\dl-curriculum.pdf")

doc = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200, 
    chunk_overlap=0,
    separator = '.')

chunks = splitter.split_documents(doc)

print(len(chunks))
print(chunks[0].page_content)
print(chunks[0].metadata)

