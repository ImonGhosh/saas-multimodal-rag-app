

<!-- Source: https://www.meilisearch.com/blog/rag-types -->

[Go to home![Meilisearch's logo](https://www.meilisearch.com/_next/static/media/logo.cd874c57.svg)](https://www.meilisearch.com/blog/</>)
  * Product
  * Solutions
  * Developers
  * [Pricing](https://www.meilisearch.com/blog/</pricing>)
  * [Blog](https://www.meilisearch.com/blog/</blog>)


[Github Repo Link: 55.7k](https://www.meilisearch.com/blog/<https:/github.com/meilisearch/meilisearch>)[Get started](https://www.meilisearch.com/blog/<https:/cloud.meilisearch.com/register>)[Request a demo](https://www.meilisearch.com/blog/<https:/meet.meilisearch.com/meetings/cloud/presentation>)
©2026 Meilisearch - All rights reserved.
[Back to articles](https://www.meilisearch.com/blog/</blog>)
# 14 types of RAG (Retrieval-Augmented Generation)
Discover 14 types of RAG (Retrieval-Augmented Generation), their uses, pros and cons, and more.
[AI](https://www.meilisearch.com/blog/</blog/tag/ai>)
02 Sept 202514 min read
[![Maya Shin](https://www.meilisearch.com/_next/image?url=https%3A%2F%2Funable-actionable-car.media.strapiapp.com%2Fdesigndisco12_9_023_b42033a19e.jpg&w=96&q=75)](https://www.meilisearch.com/blog/</author/maya>)
[Maya Shin](https://www.meilisearch.com/blog/</author/maya>)Head of Marketing @ Meilisearch[mayya_shin](https://www.meilisearch.com/blog/<https:/twitter.com/mayya_shin>)
![14 types of RAG \(Retrieval-Augmented Generation\)](https://www.meilisearch.com/_next/image?url=https%3A%2F%2Funable-actionable-car.media.strapiapp.com%2F14_types_of_rag_retrieval_augmented_generation_9e9d544a8d.png&w=3840&q=75)
Share the article
[](https://www.meilisearch.com/blog/<https:/github.com/meilisearch>)[](https://www.meilisearch.com/blog/<https:/twitter.com/intent/tweet?url=https://meilisearch.com/blog/rag-types&text=@meilisearch 14 types of RAG \(Retrieval-Augmented Generation\)>)[](https://www.meilisearch.com/blog/<https:/www.linkedin.com/shareArticle?mini=true&url=https://meilisearch.com/blog/rag-types&text=14 types of RAG \(Retrieval-Augmented Generation\)>)[](https://www.meilisearch.com/blog/<https:/www.reddit.com/submit?url=https://meilisearch.com/blog/rag-types&title=14 types of RAG \(Retrieval-Augmented Generation\)&selftext=Discover 14 types of RAG \(Retrieval-Augmented Generation\), their uses, pros and cons, and more.>)
In this article
[1. Simple RAG (original)](https://www.meilisearch.com/blog/<#1-simple-r---original>)[2. Simple RAG with memory](https://www.meilisearch.com/blog/<#2-simple-r---with-memory>)[3. Agentic RAG](https://www.meilisearch.com/blog/<#3-agentic-r->)[4. Graph RAG](https://www.meilisearch.com/blog/<#4-graph-r->)[5. Self-RAG](https://www.meilisearch.com/blog/<#5-self-r->)[6. Branched RAG](https://www.meilisearch.com/blog/<#6-branched-r->)[7. Multimodal RAG](https://www.meilisearch.com/blog/<#7-multimodal-r->)[8. Adaptive RAG](https://www.meilisearch.com/blog/<#8-adaptive-r->)[9. Speculative RAG](https://www.meilisearch.com/blog/<#9-speculative-r->)[10. Corrective RAG](https://www.meilisearch.com/blog/<#10-corrective-r->)[11. Modular RAG](https://www.meilisearch.com/blog/<#11-modular-r->)[12. Naive RAG](https://www.meilisearch.com/blog/<#12-naive-r->)[13. Advanced RAG](https://www.meilisearch.com/blog/<#13-advanced-r->)[14. HyDE (hypothetical document embedding)](https://www.meilisearch.com/blog/<#14-hy---hypothetical-document-embedding>)[Why are there different types of RAG architecture?](https://www.meilisearch.com/blog/<#why-are-there-different-types-of-r---architecture>)[What is the importance of RAG?](https://www.meilisearch.com/blog/<#what-is-the-importance-of-r-->)[What tools are used for RAG?](https://www.meilisearch.com/blog/<#what-tools-are-used-for-r-->)[Build efficient RAG systems with Meilisearch](https://www.meilisearch.com/blog/<#build-efficient-r---systems-with-meilisearch>)
In this article
RAG (retrieval-augmented generation) represents a way for AI systems to retrieve and use relevant information from external knowledge sources to generate more accurate responses.
Instead of relying only on training data, RAG uses real-time data (from documents or databases) before generating responses.
Different types of RAG architecture handle various tasks, depending on the level of complexity.
Some RAG types are simple; they retrieve once and generate an answer.
Other types require multiple steps (such as retrieving, refining, and re-generating) to improve the quality of their response.
Here is a table that briefly describes the various RAG types along with their use cases.
![RAG types.png](https://unable-actionable-car.media.strapiapp.com/RAG_types_0be7d3e99b.png)
## 1. Simple RAG (original)
Simple RAG (original) refers to the most basic form of [retrieval-augmented generation](https://www.meilisearch.com/blog/<https:/www.meilisearch.com/blog/what-is-rag>), where the AI system retrieves relevant documents from a knowledge base in a single step and uses that information to generate a response.
It is the most straightforward implementation.
![Simple RAG \(Original\).png](https://unable-actionable-car.media.strapiapp.com/Simple_RAG_Original_b58e096a23.png)
Simple RAG works by converting a user query into embeddings, searching a vector database for semantically similar content, retrieving the top matching documents, and then feeding your original question and the retrieved information to a large language model (LLM) for answer generation. The retrieval process is predictable.
Simple RAG is used in basic question answering systems, chatbots, or FAQ automation, where questions have relatively straightforward answers.
**Pros:**
  * Fast response times
  * Easy to set up and implement
  * Low computational cost


**Cons:**
  * Struggles with questions requiring multiple sources
  * No feedback after generating a response
  * Does not improve if the data retrieval is poor


## 2. Simple RAG with memory
RAG with memory refers to an enhanced version of simple RAG that can remember previous conversations.
In the context of RAG, memory refers to the AI system's ability to keep track of past interactions (such as past questions, answers, or retrieved documents). It does not just remember what was said, but understands how previous context can influence new searches.
![Simple RAG but With Memory.png](https://unable-actionable-car.media.strapiapp.com/Simple_RAG_but_With_Memory_2d48540c41.png)
Simple RAG with memory works by storing key parts of previous conversations and using them with new queries to generate better answers.
For instance, if a user asks about the capital of France and later refers to ‘its population,’ the system recalls the context to determine that the user's query is still referring to Paris.
RAG with memory is used in personal AI agents, conversational chatbots, customer support systems, or educational tutoring platforms.
**Pros:**
  * Reduces repetitive explanations
  * Encourages more human-like interactions
  * Personalizes responses based on user conversation history


**Cons:**
  * Higher processing cost than the original simple RAG
  * Higher risk of retrieving outdated or incorrect information
  * Raises questions about data privacy


## 3. Agentic RAG
[Agentic RAG](https://www.meilisearch.com/blog/<https:/www.meilisearch.com/blog/agentic-rag>) is a more dynamic RAG that acts like an experienced researcher. Instead of just retrieving the first relevant documents, it plans its approach, decides what to investigate, and then takes action using associated tools.
Agentic RAG works by breaking down a task into smaller steps. It figures out what your question needs and then searches various data sources for valuable information.
It does not stop at the first result. It checks whether what it found answers the question, and if not, it continues searching.
![Agentic RAG.png](https://unable-actionable-car.media.strapiapp.com/Agentic_RAG_bc913668a1.png)
Agentic RAG can be used in legal research where lawyers conduct comprehensive case analysis, and in financial analysis that combines market data with regulatory information. Agentic RAG is useful when you require methodical planning.
**Pros:**
  * Good for multi-step reasoning
  * Intelligent decision-making about information gathering
  * Can improve performance on complex queries


**Cons:**
  * Costs more to run due to multiple searches
  * Difficult to build and manage
  * Takes longer to respond since it is doing actual research work


## 4. Graph RAG
[Graph RAG](https://www.meilisearch.com/blog/<https:/www.meilisearch.com/blog/graph-rag>) uses a knowledge graph to understand how different pieces of information are connected. It finds relationships and patterns between pieces of data rather than just searching for matching words.
![Graph RAG.png](https://unable-actionable-car.media.strapiapp.com/Graph_RAG_b6ffce03d4.png)
Graph RAG works by mapping out how different entities in your knowledge base are interconnected. It then uses these relationships to find relevant data. Even if a document does not have your exact search terms, it might still be helpful if it is conceptually related.
Graph RAG is used in fields where the relationships between concepts are crucial, such as investigative journalism, which uncovers hidden connections, or business intelligence, which requires understanding market relationships.
**Pros:**
  * Great for complex questions requiring connecting multiple concepts
  * Helps prevent scattered answers
  * Can provide unexpected but relevant responses


**Cons:**
  * Requires significant work to build the knowledge graph
  * Slower than basic RAG systems
  * Only as good as the connections you have taught it to recognize


## 5. Self-RAG
Self-RAG is a type of RAG that improves its search by rewriting your question before looking for answers. It behaves like a researcher who constantly questions their work.
Self-RAG works by first providing an answer based on the retrieved data, then using specialized evaluation modules to check whether the answer is accurate and supported by the source material. It uses a language model to rewrite the original query, adding missing context and inferred intent from previous conversations.
![Self-RAG.png](https://unable-actionable-car.media.strapiapp.com/Self_RAG_44c6d5034a.png)
Self-RAG is used when questions are incomplete or there is insufficient detail to retrieve the proper documents.
**Pros:**
  * Catches and corrects its own mistakes before you see them
  * Helps get better results from vague questions
  * More reliable in scenarios where accuracy matters


**Cons:**
  * Higher costs to run all those extra checks
  * Slower since it is doing the work twice
  * Can be too cautious and refuse to answer when uncertain


## 6. Branched RAG
Branched RAG is a type of RAG that explores multiple lines of thought simultaneously before deciding on the best answer.
![Branched RAG.png](https://unable-actionable-car.media.strapiapp.com/Branched_RAG_a2d1bf0ffd.png)
Branched RAG works by generating responses for different interpretations of your question, retrieving answers for each one, and then comparing the answers to pick the most relevant response.
Branched RAG is used in comprehensive market research, where structured data such as technical specifications, competitor insights, and customer feedback are needed simultaneously.
**Pros:**
  * Handles open-ended questions well
  * Less likely to miss important aspects of complex questions
  * Can provide more thoughtful final response


**Cons:**
  * Complex to coordinate findings from different sources
  * Can overwhelm users with information if not properly filtered


## 7. Multimodal RAG
Multimodal RAG is a version of RAG that simultaneously uses text, images, videos, audio files, charts, and documents to answer your questions.
![Multimodal RAG.png](https://unable-actionable-car.media.strapiapp.com/Multimodal_RAG_2c097525a4.png)
[Multimodal RAG](https://www.meilisearch.com/blog/<https:/www.meilisearch.com/blog/multimodal-rag>) converts different types of content (a graph, a photo, a video clip, or a document) into a format it can search through and understand. When you ask a question, it looks through all these different media types to find relevant information and combines everything to provide an accurate response.
Multimodal RAG is used to analyze files that combine text and other forms of media.
**Pros:**
  * Works with any type of content
  * Provides complete answers using different sources
  * Great for visual topics that need multiple perspectives


**Cons:**
  * More complex to build and train
  * Requires more storage and processing power
  * Quality depends on how well it interprets various data formats


## 8. Adaptive RAG
[Adaptive RAG](https://www.meilisearch.com/blog/<https:/www.meilisearch.com/blog/adaptive-rag>) is a RAG model that learns from experience. It pays attention to what works and what doesn’t, and gradually improves its ability to respond to different kinds of questions.
![Adaptive RAG.png](https://unable-actionable-car.media.strapiapp.com/Adaptive_RAG_60242702c1.png)
Adaptive RAG works by first recognizing the type of question (simple, complex, broad, or narrow) it received, then adjusting its retrieval process and generation style based on the question to provide an accurate answer.
Adaptive RAG is used in systems that deal with all kinds of queries, such as customer support bots, research tools, and digital assistants.
**Pros:**
  * Gets better at helping you over time by learning your preferences
  * Improves relevance by adjusting to the query type
  * Can balance speed and depth when required


**Cons:**
  * It takes time to learn and improve, so early results might be inconsistent
  * More complex to build and maintain than static RAG architectures
  * Can get stuck in bad habits if it learns from poor examples


## 9. Speculative RAG
[Speculative RAG](https://www.meilisearch.com/blog/<https:/www.meilisearch.com/blog/speculative-rag>) does not wait for you to finish asking your question. Instead, it anticipates what you might want to know next and pre-fetches that information in the background.
![A graph showing how speculative RAG works. ](https://unable-actionable-car.media.strapiapp.com/Speculative_RAG_4c1f083ea9.png)
Speculative RAG works by analyzing your current question and conversation history to predict likely follow-up queries. It then retrieves relevant documents for those anticipated questions while still working on your actual question, so when you do ask that follow-up, it already has the relevant data available.
Speculative RAG is used when speed matters, such as real-time chatbots, autocomplete suggestions, or customer service systems.
**Pros:**
  * Faster response time
  * Creates a more natural conversation flow
  * Reduces waiting time when exploring complex topics


**Cons:**
  * Risk of retrieving the wrong information if the guess is inaccurate
  * Wastes computational resources on predictions that turn out wrong


## 10. Corrective RAG
[Corrective RAG](https://www.meilisearch.com/blog/<https:/www.meilisearch.com/blog/corrective-rag>) is designed to double-check its answers and correct them if something is wrong.
![Corrective RAG.png](https://unable-actionable-car.media.strapiapp.com/Corrective_RAG_fa0994f373.png)
Corrective RAG works by doing the usual search and generating an answer, but then it steps back and asks, ‘Does this fully answer the question?’
If the answer feels off, it drops the weaker sources and tries a new search to find more relevant information before updating the response.
Corrective RAG is used when accuracy is important, such as in legal research, academic writing, or policy analysis.
**Pros:**
  * Catches and fixes poor search results before you see them
  * Improves the reliability and accuracy of generated responses
  * Adds an extra layer of quality control


**Cons:**
  * Takes longer since it might need multiple search attempts
  * Can get stuck in loops if it is never satisfied with what it finds
  * Uses more computational resources performing extra searches


## 11. Modular RAG
[Modular RAG](https://www.meilisearch.com/blog/<https:/www.meilisearch.com/blog/modular-rag>) is like a toolkit: different modules handle different parts of the process, and you can combine them however you want, depending on the use case.
The system is flexible, so you can swap in a new retriever, a better reranker, or a different generator.
![Modular RAG.png](https://unable-actionable-car.media.strapiapp.com/Modular_RAG_f45a20e140.png)
Modular RAG works by breaking the system into separate components, allowing you to customize each part without rebuilding the entire system.
Modular RAG is used across multiple domain-specific research environments and production AI applications.
**Pros:**
  * Can easily optimize each component
  * Easy to upgrade or replace components without starting afresh
  * Great for customizing workflows


**Cons:**
  * More complex to coordinate all the different components
  * Takes planning ahead to figure out how all the parts will fit together


## 12. Naive RAG
Naive RAG is the simplest form of RAG. It pulls documents based on your question and passes them straight to the model without making any adjustments.
![Naive RAG.png](https://unable-actionable-car.media.strapiapp.com/Naive_RAG_2535319547.png)
Naive RAG works by converting your question into basic search terms, pulling the top documents that match those terms, and then passing them straight to the language model to generate a response. There is no filtering or reranking; it is just a simple matching algorithm.
Naive RAG is used in simple chatbots with a limited scope and basic FAQ systems where questions are predictable.
**Pros:**
  * Very simple to build and understand
  * Fast, since there is no complex processing involved
  * Low computational costs


**Cons:**
  * Struggles with complex questions
  * No verification of search results
  * Often retrieves irrelevant documents that affect the final answer


## 13. Advanced RAG
[Advanced RAG](https://www.meilisearch.com/blog/<https:/www.meilisearch.com/blog/rag-techniques>) is a more refined version of RAG that combines multiple steps (such as reranking, memory, feedback loops, branching, improved data retrieval, etc.) to get more accurate results.
![Advanced RAG.png](https://unable-actionable-car.media.strapiapp.com/Advanced_RAG_9b806ab383.png)
Advanced RAG works by layering various RAG techniques: it can rewrite the query to make it more straightforward, rank the results, check if the answer makes sense, and even review it if required, all to ensure that the generated response is the most relevant and accurate.
Advanced RAG is used in systems when making mistakes is not an option, such as in research tools or enterprise applications.
**Pros:**
  * Handles complex questions better
  * Smart enough to know which approach works best for different situations
  * Offers more control over how results are generated


**Cons:**
  * Requires expertise to build and keep running properly
  * Requires careful fine-tuning to ensure all parts work together effectively
  * Expensive to run due to its background work


## 14. HyDE (hypothetical document embedding)
HyDE is a unique approach in RAG in which the AI model starts by generating a guess as to what a good answer might look like and then uses that guess to search for real documents that match it.
![HyDe.png](https://unable-actionable-car.media.strapiapp.com/Hy_De_c9bf688f29.png)
HyDE works by generating a hypothetical answer first, converting it into a search query, and then retrieving real documents similar to the imagined one. This is the opposite of how you expect searches to work.
HyDE is used when traditional keyword searches struggle. It is helpful in academic research systems, legal databases, or medical information systems.
**Pros:**
  * Focuses on semantic meaning, not just matching terms
  * Useful in technical domains that are hard to search
  * Helpful when the original query is missing key details


**Cons:**
  * Harder to explain how it got its result
  * Slow because of the added steps
  * The hypothetical answer might lead the searcher down completely wrong paths


## Why are there different types of RAG architecture?
Different types of RAG architecture exist because no single setup works well in every situation. You choose what works best for you based on your intended use case.
Some tasks require speed and simplicity, while others call for deeper analysis, multiple sources, or even different types of input, such as images or graphs.
For example, simple RAG can handle quick and straightforward queries. But if you are working on complex research or handling messy data, you might consider something smarter, such as agentic or self-RAG.
Now, let’s see the importance of RAG.
## What is the importance of RAG?
[RAG is a technique](https://www.meilisearch.com/blog/<https:/www.meilisearch.com/blog/rag-techniques>) that combines information retrieval with generative AI. It retrieves documents from a knowledge base and uses them to generate a more relevant and accurate answer.
It is important because it helps language models stay grounded in true information rather than hallucinations.
RAGs are widely used for various real-world use cases.
Tech companies, for instance, use RAG to help support agents quickly find answers to technical issues without going through pages of documentation.
Law firms use it to scan thousands of legal documents in seconds.
Hospitals also use RAG to enable doctors to match patient symptoms with findings from medical literature, thereby improving their diagnosis and treatment decisions.
## What tools are used for RAG?
RAG tools are the building blocks that connect your data to powerful language models to deliver accurate results.
Here are some [retrieval-augmented generation tools](https://www.meilisearch.com/blog/<https:/www.meilisearch.com/blog/rag-tools>) that are used in different RAG applications:
  * **Meilisearch:** A super-fast search engine that supports both keyword and vector search. It is excellent for balancing keyword and semantic search and works smoothly with embedding models through a simple API.
  * **LangChain:** An open-source orchestration framework that connects retrievers, embedding models, and generators. It helps manage your entire RAG pipeline and handles the process of integrating with APIs, databases, and file formats.
  * **Weaviate:** A vector database built for production. It supports hybrid search, filtering, and fast, scalable queries.
  * **Faiss:** A vector search library developed by Meta. It allows you to index and search embeddings efficiently, ensuring that the AI system focuses on meaning instead of just keywords.
  * **Haystack:** A complete RAG framework that combines retrieval, question answering, and generation. It is useful when you want all RAG components to work together seamlessly.


RAG tools are categorized based on the part of the pipeline they support (retrieval, generation process, orchestration, or storage). Choosing the right tool depends on what stage you are working on and what you are trying to build.
## Build efficient RAG systems with Meilisearch
Choosing the right RAG approach comes down to matching the architecture to the problem you’re solving.
As the technology evolves, RAG will continue to play a central role in building AI systems that stay grounded in relevant data.
Combining strong retrieval with effective generative models helps reduce errors, improve reliability, and make AI more useful in real-world scenarios – whether for quick lookups, complex research, or anything in between.
#### Pick a RAG tool that suits your needs
Meilisearch is a lightning-fast, open-source search engine that has become a favorite for building RAG systems. Unlike traditional databases that can be slow when searching through large amounts of text, Meilisearch is explicitly built for speed and can handle typos, partial matches, and complex queries.
[Start building with Meilisearch Cloud today](https://www.meilisearch.com/blog/<https:/www.meilisearch.com/cloud>)
[![Knowledge graph vs. vector database for RAG: which is best?](https://www.meilisearch.com/_next/image?url=https%3A%2F%2Funable-actionable-car.media.strapiapp.com%2FKnowledge_graph_vs_vector_database_for_RAG_fdbdb3a39e.png&w=1200&q=75)](https://www.meilisearch.com/blog/</blog/knowledge-graph-vs-vector-database-for-rag>)
[AI](https://www.meilisearch.com/blog/</blog/tag/ai>)
## [Knowledge graph vs. vector database for RAG: which is best?](https://www.meilisearch.com/blog/</blog/knowledge-graph-vs-vector-database-for-rag>)
Learn the key differences between knowledge graphs and vector databases for RAG, when to use each, and how to combine them for optimal results.
[![Maya Shin](https://www.meilisearch.com/_next/image?url=https%3A%2F%2Funable-actionable-car.media.strapiapp.com%2Fdesigndisco12_9_023_b42033a19e.jpg&w=96&q=75)](https://www.meilisearch.com/blog/</author/maya>)
[Maya Shin](https://www.meilisearch.com/blog/</author/maya>)15 Jan 2026
[![Retrieval-Augmented Generation \(RAG\) for business: Full guide](https://www.meilisearch.com/_next/image?url=https%3A%2F%2Funable-actionable-car.media.strapiapp.com%2FRAG_for_Business_Full_Guide_e97c373bca.png&w=1200&q=75)](https://www.meilisearch.com/blog/</blog/rag-for-business>)
[AI](https://www.meilisearch.com/blog/</blog/tag/ai>)
## [Retrieval-Augmented Generation (RAG) for business: Full guide](https://www.meilisearch.com/blog/</blog/rag-for-business>)
Explore how RAG for business boosts AI accuracy and delivers smarter, context-driven insights.
[![Maya Shin](https://www.meilisearch.com/_next/image?url=https%3A%2F%2Funable-actionable-car.media.strapiapp.com%2Fdesigndisco12_9_023_b42033a19e.jpg&w=96&q=75)](https://www.meilisearch.com/blog/</author/maya>)
[Maya Shin](https://www.meilisearch.com/blog/</author/maya>)06 Jan 2026
[![Understanding hybrid search RAG for better AI answers](https://www.meilisearch.com/_next/image?url=https%3A%2F%2Funable-actionable-car.media.strapiapp.com%2FUnderstanding_hybrid_search_RAG_for_better_AI_answers_8b9412d6fd.png&w=1200&q=75)](https://www.meilisearch.com/blog/</blog/hybrid-search-rag>)
[AI](https://www.meilisearch.com/blog/</blog/tag/ai>)[State of search](https://www.meilisearch.com/blog/</blog/tag/state-of-search>)
## [Understanding hybrid search RAG for better AI answers](https://www.meilisearch.com/blog/</blog/hybrid-search-rag>)
Learn what hybrid search RAG is, how it blends semantic and keyword search for more accurate retrieval, and how it works, challenges, implementation & more.
[![Maya Shin](https://www.meilisearch.com/_next/image?url=https%3A%2F%2Funable-actionable-car.media.strapiapp.com%2Fdesigndisco12_9_023_b42033a19e.jpg&w=96&q=75)](https://www.meilisearch.com/blog/</author/maya>)
[Maya Shin](https://www.meilisearch.com/blog/</author/maya>)16 Dec 2025
Older posts
Newsletter
Subscribe and never miss out on our latest release, blog posts and news.
Submit
Product
  * [Meilisearch Cloud](https://www.meilisearch.com/blog/</cloud>)
  * [Pricing](https://www.meilisearch.com/blog/</pricing>)
  * [Documentation](https://www.meilisearch.com/blog/</docs>)
  * [Release notes](https://www.meilisearch.com/blog/<https:/updates.cloud.meilisearch.com/>)
  * [Comparisons](https://www.meilisearch.com/blog/</docs/learn/what_is_meilisearch/comparison_to_alternatives>)
  * [Trust center](https://www.meilisearch.com/blog/<https:/trust.meilisearch.com>)
  * [Launch week](https://www.meilisearch.com/blog/<https:/www.meilisearch.com/launch-week-q3-2025>)


Contact us
  * [GitHub](https://www.meilisearch.com/blog/<https:/github.com/meilisearch>)
  * [Discord](https://www.meilisearch.com/blog/<https:/discord.meilisearch.com/>)
  * [Newsletter](https://www.meilisearch.com/blog/<https:/share-eu1.hsforms.com/1LN5N0x_GQgq7ss7tXmSykwfg3aq>)
  * [Helpdesk](https://www.meilisearch.com/blog/<https:/help.meilisearch.com>)


Company
  * [Blog](https://www.meilisearch.com/blog/<https:/meilisearch.com/blog>)
  * [Careers](https://www.meilisearch.com/blog/</careers>)
  * [Customers](https://www.meilisearch.com/blog/<https:/www.meilisearch.com/customers>)
  * [Code of conduct](https://www.meilisearch.com/blog/<https:/github.com/meilisearch/MeiliSearch/blob/main/CODE_OF_CONDUCT.md>)
  * [Privacy policy](https://www.meilisearch.com/blog/</privacy-policy>)
  * [Terms of use](https://www.meilisearch.com/blog/</terms-of-use>)
  * [Swag store](https://www.meilisearch.com/blog/<https:/swag.meilisearch.com>)


  * [](https://www.meilisearch.com/blog/<https:/github.com/meilisearch>)


  * [](https://www.meilisearch.com/blog/<https:/twitter.com/meilisearch/>)


  * [](https://www.meilisearch.com/blog/<https:/www.linkedin.com/company/meilisearch>)


  * [](https://www.meilisearch.com/blog/<https:/discord.meilisearch.com/>)


©2026 Meilisearch - All rights reserved.
  * [](https://www.meilisearch.com/blog/<https:/github.com/meilisearch>)


  * [](https://www.meilisearch.com/blog/<https:/twitter.com/meilisearch/>)


  * [](https://www.meilisearch.com/blog/<https:/www.linkedin.com/company/meilisearch>)


  * [](https://www.meilisearch.com/blog/<https:/discord.meilisearch.com/>)



