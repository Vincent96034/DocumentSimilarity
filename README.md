# Document Similarity

## Task Outline

The primary objective of this project is to develop a Python program capable of computing text document similarity between various documents. This implementation involves processing a list of documents as the input text corpus and constructing a dictionary of words for the corpus. Subsequently, upon receiving a new document (search document), the program will output a list of documents similar to the search document, ranked in descending order of similarity.

## Design Philosophy

The design philosophy centers on creating a modular, easily extendible system, prioritizing encapsulation of features. This involves:

- **Extensibility of functionalities:** The code architecture is built around extendability. Adding new embedding algorithms, vector databases, or similarity measures in the future should be very simple.
- **Separation of User Interface and Implementation Interface:** Ensuring that the implementation details are abstracted behind a user-friendly interface. Differentiating between internal methods for core functionality and public endpoints for user interaction.

## Project Structure

The project is structured into several key components divided into separate python modules. To use them, one simply can import the specific objects into a file.

### Embedding Module

This module contains the base class blueprint for all embedding algorithms. In this project, a simple one-hot encoding algorithm, as well as a more sophisticated embedding method from gensim (doc2vec), was implemented.

- **`embedder.py`** -- Abstract base class for embeddings.
- **`doc2vecembedder.py`** -- Implementation of Doc2Vec embedding.
- **`onehotembedder.py`** -- Implementation of one-hot encoding for embedding.
- **`stopwords.py`** -- Handling stopwords in document processing.

### Models

This module contains general objects and models this implementation uses. This includes a new definition of a vector object to store embeddings, as well as text data or metadata, and classes to define text documents in general.

- **`text.py`** -- Text models for document processing.
- **`_types.py`** -- Custom types for model structuring.
- **`vec.py`** -- Vector model for document representation.

### VectorStore Module

This module contains the functionality regarding the storage of vectors and finding similar vectors in such storage objects. There is a base class blueprint of a vector-database and a simple implementation of such a database. This could be extended with a real production-capable database like Pinecone, for example. Additionally, this module contains different similarity measures (all built from a parent blueprint class) and a respective measure factory to extract the specific measure when needed.

- **`vectordb.py`** -- Base class for vector database handling.
- **`simplevec.py`** -- Simple vector implementation.
- **`measurefactory`** -- Factory pattern for creating measure instances.
- **Measures**:
  - **`cosine`**, **`dot`**, **`euclidean`** -- Various measures for calculating document similarity.
  - **`measure`** -- Base class for similarity measures.

## Embeddings

The architecture of the embedding component within the codebase follows a modular and hierarchical design. At the top level, we have the `embedding` abstract class which serves as a blueprint for all embedding implementations. This design allows for a consistent interface across different embedding strategies, ensuring that each specific model conforms to a standard set of operations required for document embedding. The public methods like `fit()` or `embed()` in the abstract class serve as the exposed interface for interaction with the embedding models. They define the contract that all concrete implementations must follow, allowing users to employ different embedding strategies interchangeably. User-input is handled only in the public methods and then transformed into the standardized format for the implementations. This design choice follows the principle of abstraction, promoting a user-friendly experience while safeguarding the integrity of the data processing carried out by the private methods.

Underneath the abstract class, we introduce `text models`, which encapsulate the representation of text data. These models are crucial as they provide a structured and standardized way to manage the text data we wish to embed, allowing for flexibility and ease of manipulation across various embedding techniques.

The implementations are concrete classes that extend the abstract `embedding` class. Here, we have specific strategies like `doc2vecembedder` and `onehotembedder`. Each implementation follows the interface defined by the `embedding` class but provides a unique approach to text representation. The decision to separate the implementations from the abstract class is guided by the principle of single responsibility and open/closed principle, aiming for a codebase that is easy to extend but closed for modification of existing code, facilitating maintenance and future enhancements.

## Similarity Search

The main objective of this project is to find similar text documents, embedded as vectors. We use a structured approach to effectively search and compare document vectors.

### Vector Storage

The backbone of similarity search is the database abstract class. This class provides a general

 framework for storing and retrieving vector representations of documents, ensuring a uniform interface regardless of the underlying storage mechanism. Implementations of this class may vary, offering different trade-offs in terms of performance and scalability, thereby allowing the system to adapt to various storage requirements. We implemented a very simple vector database ourselves, but more sophisticated implementations like PineconeDB could easily be implemented into the predefined structure. Similar to the embedding abstract class, only the private methods must be implemented to conform to the project structure.

### Vector Model

The vector model is central to the system, defining key operations like addition and subtraction on vectors. These operations are fundamental to manipulating vector representations, which is crucial in computing similarities between documents. The decision to define these operations within the vector model follows the principle of encapsulation, ensuring that all vector-related operations are centralized and easily manageable. A vector in this implementation can include the actual text data as well as additional metadata. In the simple vector database, those vectors are essentially stored in a dictionary to query from.

### Measures Abstract Class

The measures abstract class outlines the structure for various similarity metrics, such as cosine similarity, dot product, and Euclidean distance. Each implementation of this class provides a different method to quantify the similarity between document vectors, catering to different types of data and similarity criteria. By abstracting these measures, the system gains the flexibility to easily introduce new similarity metrics as needed. Each measure must implement the `calculate` function as well as the `sort_func`. The latter defines how to sort according to similarity score; this might differ from implementation to implementation.

### Similarity Search

In the similarity search process, the choice of similarity measure can significantly impact the results. For instance, cosine similarity is often preferred for text data as it is length invariant, making it suitable for documents of varying lengths. In contrast, Euclidean distance might be more effective in dense vector spaces. The system's design accommodates these variations by allowing easy experimentation with different measures.

## How to use
The provided notebook `main.ipynb` will guide you on how to use the code.
