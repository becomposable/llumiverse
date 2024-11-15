import { GenerateContentRequest, VertexAI } from "@google-cloud/vertexai";
import { AIModel, AbstractDriver, Completion, CompletionChunkObject, DriverOptions, EmbeddingsResult, ExecutionOptions, ModelSearchPayload, PromptOptions, PromptSegment } from "@llumiverse/core";
import { FetchClient } from "api-fetch-client";
import { GoogleAuth, GoogleAuthOptions } from "google-auth-library";
import { JSONClient } from "google-auth-library/build/src/auth/googleauth.js";
import { TextEmbeddingsOptions, getEmbeddingsForText } from "./embeddings/embeddings-text.js";
import { BuiltinModels, getModelDefinition } from "./models.js";
import { EmbeddingsOptions } from "@llumiverse/core";
import { getEmbeddingsForImages } from "./embeddings/embeddings-image.js";


export interface VertexAIDriverOptions extends DriverOptions {
    project: string;
    region: string;
    googleAuthOptions?: GoogleAuthOptions;
}

export class VertexAIDriver extends AbstractDriver<VertexAIDriverOptions, GenerateContentRequest> {
    static PROVIDER = "vertexai";
    provider = VertexAIDriver.PROVIDER;

    //aiplatform: v1.ModelServiceClient;
    vertexai: VertexAI;
    fetchClient: FetchClient;
    authClient: JSONClient | GoogleAuth<JSONClient>;

    constructor( options: VertexAIDriverOptions) {
        super(options);
        //this.aiplatform = new v1.ModelServiceClient();

        this.authClient = options.googleAuthOptions?.authClient ?? new GoogleAuth(options.googleAuthOptions);

        this.vertexai = new VertexAI({
            project: this.options.project,
            location: this.options.region,
            googleAuthOptions: this.options.googleAuthOptions,
        });
        this.fetchClient = createFetchClient({
            region: this.options.region,
            project: this.options.project,
        }).withAuthCallback(async () => {
            //@ts-ignore
            const token = await this.authClient.getAccessToken();
            return `Bearer ${token}`;
        });
        // this.aiplatform = new v1.ModelServiceClient({
        //     projectId: this.options.project,
        //     apiEndpoint: `${this.options.region}-${API_BASE_PATH}`,
        // });
    }

    protected canStream(options: ExecutionOptions): Promise<boolean> {
        return Promise.resolve(getModelDefinition(options.model).model.can_stream === true);
    }

    public createPrompt(segments: PromptSegment[], options: PromptOptions): Promise<GenerateContentRequest> {
        return getModelDefinition(options.model).createPrompt(this, segments, options);
    }

    async requestCompletion(prompt: GenerateContentRequest, options: ExecutionOptions): Promise<Completion<any>> {
        return getModelDefinition(options.model).requestCompletion(this, prompt, options);
    }
    async requestCompletionStream(prompt: GenerateContentRequest, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        return getModelDefinition(options.model).requestCompletionStream(this, prompt, options);
    }

    async listModels(_params?: ModelSearchPayload): Promise<AIModel<string>[]> {
        return BuiltinModels;
        // try {
        //     const response = await this.fetchClient.get('/publishers/google/models/gemini-pro');
        //     console.log('>>>>>>>>', response);
        // } catch (err: any) {
        //     console.error('+++++VETREX ERROR++++++', err);
        //     throw err;
        // }

        // TODO uncomment this to use apiplatform instead of the fetch client
        // const response = await this.aiplatform.listModels({
        //     parent: `projects/${this.options.project}/locations/${this.options.region}`,
        // });

        return []; //TODO
    }

    validateConnection(): Promise<boolean> {
        throw new Error("Method not implemented.");
    }

    async generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        if (options.image || options.model?.includes("multimodal")) {
            if (options.text && options.image) {
                throw new Error("Text and Image simultaneous embedding not implemented. Submit seperately");
            }
            return getEmbeddingsForImages(this, options);
        }
        const text_options: TextEmbeddingsOptions = {
            content: options.text ?? '',
            model: options.model,
        }
        return getEmbeddingsForText(this, text_options);
    }

}

//'us-central1-aiplatform.googleapis.com',
const API_BASE_PATH = 'aiplatform.googleapis.com';
function createFetchClient({ region, project, apiEndpoint, apiVersion = 'v1' }: {
    region: string;
    project: string;
    apiEndpoint?: string;
    apiVersion?: string;
}) {
    const vertexBaseEndpoint = apiEndpoint ?? `${region}-${API_BASE_PATH}`;
    return new FetchClient(`https://${vertexBaseEndpoint}/${apiVersion}/projects/${project}/locations/${region}`).withHeaders({
        'Content-Type': 'application/json',
    });
}
