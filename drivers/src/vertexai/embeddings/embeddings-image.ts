
import { EmbeddingsOptions, EmbeddingsResult } from '@llumiverse/core';
import { VertexAIDriver } from '../index.js';

export interface ImageEmbeddingsOptions {
    model?: string;
    image?: {bytesBase64Encoded?: string}, // the image to generate embeddings for
    text: string,                           // the text to generate embeddings for
}

interface EmbedingsForImagePrompt {
    instances: ImageEmbeddingsOptions[]
}

interface ImageEmbeddingsResult {
    predictions: [
        {
            imageEmbedding: number[]
            textEmbedding: number[]
        }
    ],
    deployedModelId: string,
}

//Currently we are only supporting either text or images sent to the multimodal model.
export async function getEmbeddingsForImages(driver: VertexAIDriver, options: EmbeddingsOptions): Promise<EmbeddingsResult> {

    // API is returns a 400 Error if a property is empty, so you undefined and "as" to remove the property entirely.
    const prompt = {
        instances: [{
            text: options.image ? undefined : options.text,
            image: options.image ?
                {
                    bytesBase64Encoded: options.image
                }
                : undefined,
        }]
    } as EmbedingsForImagePrompt;

    const model = options.model || "multimodalembedding@001";

    const result = await driver.fetchClient.post(`/publishers/google/models/${model}:predict`, {
        payload: prompt
    }) as ImageEmbeddingsResult;

    return {
        values: result.predictions[0].imageEmbedding ?? result.predictions[0].textEmbedding,
        model: model,
    };
}
