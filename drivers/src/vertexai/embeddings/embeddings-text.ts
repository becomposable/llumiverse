
import { EmbeddingsResult } from '@llumiverse/core';
import { VertexAIDriver } from '../index.js';

export interface TextEmbeddingsOptions {
    model?: string;
    task_type?: "RETRIEVAL_QUERY" | "RETRIEVAL_DOCUMENT" | "SEMANTIC_SIMILARITY" | "CLASSIFICATION" | "CLUSTERING",
    title?: string, // the title for the embedding
    content: string // the text to generate embeddings for
}

interface EmbedingsForTextPrompt {
    instances: TextEmbeddingsOptions[]
}

interface TextEmbeddingsResult {
    predictions: [
        {
            embeddings: TextEmbeddings
        }
    ]
}

interface TextEmbeddings {
    statistics: {
        truncated: boolean,
        token_count: number
    },
    values: [number]
}

export async function getEmbeddingsForText(driver: VertexAIDriver, options: TextEmbeddingsOptions): Promise<EmbeddingsResult> {
    const prompt = {
        instances: [{
            task_type: options.task_type,
            title: options.title,
            content: options.content
        }]
    } as EmbedingsForTextPrompt;

    const model = options.model || "text-embedding-004";

    const result = await driver.fetchClient.post(`/publishers/google/models/${model}:predict`, {
        payload: prompt
    }) as TextEmbeddingsResult;

    return {
        ...result.predictions[0].embeddings,
        model,
        token_count: result.predictions[0].embeddings.statistics?.token_count
    };
}
