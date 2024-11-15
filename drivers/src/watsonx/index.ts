import { AIModel, AbstractDriver, Completion, DriverOptions, EmbeddingsOptions, EmbeddingsResult, ExecutionOptions, CompletionChunk } from "@llumiverse/core";
import { transformSSEStream } from "@llumiverse/core/async";
import { FetchClient } from "api-fetch-client";
import { GenerateEmbeddingPayload, GenerateEmbeddingResponse, WatsonAuthToken, WatsonxListModelResponse, WatsonxModelSpec, WatsonxTextGenerationPayload, WatsonxTextGenerationResponse } from "./interfaces.js";

interface WatsonxDriverOptions extends DriverOptions {
    apiKey: string;
    projectId: string;
    endpointUrl: string;
}

const API_VERSION = "2024-03-14"

export class WatsonxDriver extends AbstractDriver<WatsonxDriverOptions, string> {
    static PROVIDER = "watsonx";
    provider = WatsonxDriver.PROVIDER;
    apiKey: string;
    endpoint_url: string;
    projectId: string;
    authToken?: WatsonAuthToken;
    fetcher?: FetchClient;
    fetchClient: FetchClient

    constructor(options: WatsonxDriverOptions) {
        super(options);
        this.apiKey = options.apiKey;
        this.projectId = options.projectId;
        this.endpoint_url = options.endpointUrl;
        this.fetchClient = new FetchClient(this.endpoint_url).withAuthCallback(async () => this.getAuthToken().then(token => `Bearer ${token}`));
    }

    async requestCompletion(prompt: string, options: ExecutionOptions): Promise<Completion<any>> {
        const payload: WatsonxTextGenerationPayload = {
            model_id: options.model,
            input: prompt + "\n",
            parameters: {
                max_new_tokens: options.max_tokens,
                //time_limit: options.time_limit,
            },
            project_id: this.projectId,
        }

        const res = await this.fetchClient.post(`/ml/v1/text/generation?version=${API_VERSION}`, { payload }) as WatsonxTextGenerationResponse;

        const result = res.results[0];

        return {
            result: result.generated_text,
            token_usage: {
                prompt: result.input_token_count,
                result: result.generated_token_count,
                total: result.input_token_count + result.generated_token_count,
            },
            finish_reason: watsonFinishReason(result.stop_reason),
            original_response: options.include_original_response ? res : undefined,
        }
    }

    async requestCompletionStream(prompt: string, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunk>> {

        const payload: WatsonxTextGenerationPayload = {
            model_id: options.model,
            input: prompt + "\n",
            parameters: {
                max_new_tokens: options.max_tokens,
                //time_limit: options.time_limit,
            },
            project_id: this.projectId,
        }

        const stream = await this.fetchClient.post(`/ml/v1/text/generation_stream?version=${API_VERSION}`, {
            payload: payload,
            reader: 'sse'
        })

        return transformSSEStream(stream, (data: string) => {
            const json = JSON.parse(data) as WatsonxTextGenerationResponse;
            return {
                result: json.results[0]?.generated_text ?? '',
                finish_reason: watsonFinishReason(json.results[0]?.stop_reason),
                token_usage: {
                    prompt: json.results[0].input_token_count,
                    result: json.results[0].generated_token_count,
                    total: json.results[0].input_token_count + json.results[0].generated_token_count,
                },
            };
        });

    }



    async listModels(): Promise<AIModel<string>[]> {



        const res = await this.fetchClient.get(`/ml/v1/foundation_model_specs?version=${API_VERSION}`)
            .catch(err => this.logger.warn("Can't list models on Watsonx: " + err)) as WatsonxListModelResponse;

        const aimodels = res.resources.map((m: WatsonxModelSpec) => {
            return {
                id: m.model_id,
                name: m.label,
                description: m.short_description,
                provider: this.provider,
            }
        });

        return aimodels;

    }

    async getAuthToken(): Promise<string> {


        if (this.authToken) {
            const now = Date.now() / 1000;
            if (now < this.authToken.expiration) {
                return this.authToken.access_token;
            } else {
                this.logger.debug("Token expired, refetching", this.authToken, now)
            }
        }
        const authToken = await fetch('https://iam.cloud.ibm.com/identity/token', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=${this.apiKey}`,
        }).then(response => response.json()) as WatsonAuthToken;

        this.authToken = authToken;
        return this.authToken.access_token;

    }

    async validateConnection(): Promise<boolean> {
        return this.listModels()
            .then(() => true)
            .catch((err) => {
                this.logger.warn("Failed to connect to WatsonX", err);
                return false
            });
    }

    async generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        if (options.image) {
            throw new Error("Image embeddings not supported by Watsonx");
        }

        if (!options.text) {
            throw new Error ("No text provided");
        }

        const payload: GenerateEmbeddingPayload = {
            inputs: [options.text],
            model_id: options.model ?? 'ibm/slate-125m-english-rtrvr',
            project_id: this.projectId
        }

        const res = await this.fetchClient.post(`/ml/v1/text/embeddings?version=${API_VERSION}`, { payload }) as GenerateEmbeddingResponse;

        return {
            values: res.results[0].embedding,
            model: res.model_id
        }

    }

}

function watsonFinishReason(reason: string | undefined) {
    if (!reason) return undefined;
    switch (reason) {
        case 'eos_token': return "stop";
        case 'max_tokens': return "length";
        default: return reason;
    }
}



/*interface ListModelsParams extends ModelSearchPayload {
    limit?: number;
}*/
