import "dotenv/config";
import express from "express";
import cors from "cors";
import { CohereClient } from "cohere-ai";
import fs from "fs/promises";
import path from "path";

const app = express();
app.use(cors());
app.use(express.json());

const cohere = new CohereClient({
  token: process.env.COHERE_API_KEY,
});

const EMBEDDINGS_FILE = "./documents/embeddings.json";

function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((sum, a, idx) => sum + a * vecB[idx], 0);
  const magA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (magA * magB);
}

function getTopKDocuments(queryEmbedding, documents, k = 5) {
  const similarities = documents.map((doc) => ({
    doc,
    score: cosineSimilarity(queryEmbedding, doc.embedding),
  }));

  similarities.sort((a, b) => b.score - a.score);
  return similarities.slice(0, k).map((item) => item.doc);
}

async function loadDocuments() {
  const files = [
    "./documents/api_patterns.json",
    "./documents/architectures.json",
    "./documents/design_patterns.json",
    "./documents/dsa.json",
    "./documents/languages.json",
    "./documents/projects.json",
  ];

  const documents = [];

  for (const file of files) {
    try {
      const content = await fs.readFile(file, "utf-8");
      const json = JSON.parse(content);

      // Handler for api_patterns.json
      if (json.api_styles) {
        json.api_styles.forEach((style) => {
          documents.push({
            id: `api_${style.name.replace(/\s+/g, "_")}`,
            data: {
              title: `${style.name} API Style`,
              snippet: `${
                style.description
              } Best use cases: ${style.best_use_cases.join(
                ", "
              )}. Strengths include: ${style.strengths.join(", ")}.`,
            },
          });
        });
      }
      // Handler for architectures.json
      else if (json.architectures) {
        json.architectures.forEach((arch) => {
          documents.push({
            id: `arch_${arch.architecture_name.replace(/\s+/g, "_")}`,
            data: {
              title: `${arch.architecture_name} Architecture`,
              snippet: `${
                arch.description
              } This architecture is suitable for projects like ${arch.project_types.join(
                ", "
              )}. Key strengths are: ${arch.strengths.join(", ")}.`,
            },
          });
        });
      }
      // Handler for design_patterns.json
      else if (json.design_patterns) {
        json.design_patterns.forEach((category) => {
          category.patterns.forEach((pattern) => {
            documents.push({
              id: `pattern_${pattern.name.replace(/\s+/g, "_")}`,
              data: {
                title: `${pattern.name} (Design Pattern)`,
                snippet: `Category: ${category.category}. Purpose: ${pattern.description}. Use this pattern when you want to solve this problem: ${pattern.problem_it_solves}`,
              },
            });
          });
        });
      }
      // Handler for dsa.json (Data Structures and Algorithms)
      else if (json.data_structures_and_algorithms) {
        json.data_structures_and_algorithms.forEach((item) => {
          const exampleProject = item.project_applications[0];
          documents.push({
            id: `dsa_${item.name.replace(/\s+/g, "_")}`,
            data: {
              title: `Data Structure/Algorithm: ${item.name}`,
              snippet: `${item.description} It is typically used for: ${item.how_it_is_used}. A sample project is a ${exampleProject.project}, where it is applied like this: ${exampleProject.application}`,
            },
          });
        });
      }
      // Handler for languages.json
      else if (json.programming_languages) {
        json.programming_languages.forEach((lang) => {
          documents.push({
            id: `lang_${lang.name.replace(/\s+/g, "_")}`,
            data: {
              title: `Programming Language: ${lang.name}`,
              snippet: `Known for: ${lang.known_for}. Ideal for ${
                lang.ideal_for
              }. Strengths: ${lang.strengths.join(
                ", "
              )}. Paradigms: ${lang.paradigms.join(", ")}. Typing: ${
                lang.typing
              }.`,
            },
          });
        });
      }
      // Handler for the main projects.json file
      else if (Array.isArray(json) && json[0]?.project_ideas) {
        json.forEach((category) => {
          category.project_ideas.forEach((project) => {
            documents.push({
              id: `project_${project.name.replace(/\s+/g, "_")}`,
              data: {
                title: `Project Idea: ${project.name}`,
                snippet: `Category: ${category.category}. Description: ${
                  project.description
                } Suitable languages include: ${project.languages.join(
                  ", "
                )}. Difficulty: ${project.difficulty}.`,
              },
            });
          });
        });
      }
    } catch (err) {
      console.error(`Error reading or parsing ${file}:`, err);
    }
  }

  console.log(`Loaded ${documents.length} documents from all files.`);
  return documents;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function embedDocumentsInBatches(documents, batchSize = 96) {
  const allEmbeddings = [];

  for (let i = 0; i < documents.length; i += batchSize) {
    const batch = documents.slice(i, i + batchSize);
    console.log(
      `Embedding batch ${i / batchSize + 1} of ${Math.ceil(
        documents.length / batchSize
      )}...`
    );

    const response = await cohere.embed({
      texts: batch.map((doc) => `${doc.data.title}. ${doc.data.snippet}`),
      model: "embed-multilingual-v3.0",
      input_type: "search_document",
    });

    allEmbeddings.push(...response.embeddings);

    // Delay between batches to avoid 429
    await sleep(10000);
  }

  return allEmbeddings;
}

async function saveEmbeddingsToFile(documents) {
  const embeddingsData = documents.map((doc) => ({
    id: doc.id,
    title: doc.data.title,
    snippet: doc.data.snippet,
    embedding: doc.embedding,
  }));

  await fs.writeFile(
    EMBEDDINGS_FILE,
    JSON.stringify(embeddingsData, null, 2),
    "utf-8"
  );
  console.log(`Embeddings saved to ${EMBEDDINGS_FILE}`);
}

async function loadEmbeddingsFromFile() {
  try {
    const content = await fs.readFile(EMBEDDINGS_FILE, "utf-8");
    const embeddingsData = JSON.parse(content);
    console.log(
      `Loaded ${embeddingsData.length} embeddings from ${EMBEDDINGS_FILE}`
    );
    return embeddingsData.map((item) => ({
      id: item.id,
      data: { title: item.title, snippet: item.snippet },
      embedding: item.embedding,
    }));
  } catch (err) {
    console.warn("No existing embeddings file found. Will compute embeddings.");
    return null;
  }
}

let cachedDocuments = [];

async function initializeDocuments() {
  cachedDocuments = await loadEmbeddingsFromFile();
  if (!cachedDocuments) {
    const documents = await loadDocuments();

    console.log("Embedding documents...");
    const allEmbeddings = await embedDocumentsInBatches(documents);

    allEmbeddings.forEach((embedding, i) => {
      documents[i].embedding = embedding;
    });

    await saveEmbeddingsToFile(documents);
    cachedDocuments = documents;
    console.log("Embeddings ready.");
  }
}

// Initialize on server start
initializeDocuments();

app.post("/generate", async (req, res) => {
  const { prompt } = req.body;

  if (!prompt) {
    return res.status(400).json({ error: "Prompt is required" });
  }

  try {
    console.log(`Embedding user prompt...`);
    const embedResponse = await cohere.embed({
      texts: [prompt],
      model: "embed-multilingual-v3.0",
      input_type: "search_document",
    });

    const queryEmbedding = embedResponse.embeddings[0];

    const topDocuments = getTopKDocuments(queryEmbedding, cachedDocuments, 10);

    console.log(`Retrieved top ${topDocuments.length} documents.`);

    const response = await cohere.chat({
      model: "command-r-plus",
      message: prompt,
      documents: topDocuments.map((doc) => ({
        text: `${doc.data.title}. ${doc.data.snippet}`,
      })),
      preamble:
        "You are a professional and friendly coding assistant.  You must answer the users questions using ONLY the information provided in the documents below whenever possible. If a topic is not covered by the documents, you may use your own knowledge — but ONLY in the domain of programming and project architectures. Stay strictly within this domain: programming,programming languges, best practices in architectures patterns, design patterns, project types, data structures, algorithms, fun projects, project based learning and things to do. DO NOT provide information about spagethi code, bad practices, cheap work arounds, innefficient patterns or algorithims. Always write in a helpful, engaging tone,  and it may be more technical and comprehensive for developers of all experience levels. If language or pattern not refered to in documents please tell user to give more insight",
      temperature: 0.4,
    });

    console.log("Cohere chat response:", JSON.stringify(response, null, 2));

    res.json({
      text: response.text,
      citations: response.citations ?? [],
    });
  } catch (err) {
    console.error("Error communicating with Cohere API:", err);
    res.status(500).json({ error: "Cohere request failed" });
  }
});


app.listen(5000, () => {
  console.log("Listening on http://localhost:5000");
});
