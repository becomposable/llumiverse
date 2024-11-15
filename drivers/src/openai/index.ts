import {
    AIModel,
    AbstractDriver,
    Completion,
    DataSource,
    DriverOptions,
    EmbeddingsOptions,
    EmbeddingsResult,
    ExecutionOptions,
    ExecutionTokenUsage,
    ModelType,
    CompletionChunkObject,
    TrainingJob,
    TrainingJobStatus,
    TrainingOptions,
    TrainingPromptOptions
} from "@llumiverse/core";
import { asyncMap } from "@llumiverse/core/async";
import { formatOpenAILikeMultimodalPrompt } from "@llumiverse/core/formatters";
import OpenAI, { AzureOpenAI } from "openai";
import { Stream } from "openai/streaming";

const supportFineTunning = new Set([
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0613",
    "babbage-002",
    "davinci-002",
    "gpt-4-0613"
]);

export interface BaseOpenAIDriverOptions extends DriverOptions {
}

export abstract class BaseOpenAIDriver extends AbstractDriver<
    BaseOpenAIDriverOptions,
    OpenAI.Chat.Completions.ChatCompletionMessageParam[]
> {
    abstract provider: "azure_openai" | "openai";
    abstract service: OpenAI | AzureOpenAI ;

    constructor(opts: BaseOpenAIDriverOptions) {
        super(opts);
        this.formatPrompt = formatOpenAILikeMultimodalPrompt as any //TODO: better type, we send back OpenAI.Chat.Completions.ChatCompletionMessageParam[] but just not compatbile with Function call that we don't use here

    }

    extractDataFromResponse(
        options: ExecutionOptions,
        result: OpenAI.Chat.Completions.ChatCompletion
    ): Completion {
        const tokenInfo: ExecutionTokenUsage = {
            prompt: result.usage?.prompt_tokens,
            result: result.usage?.completion_tokens,
            total: result.usage?.total_tokens,
        };

        const choice = result.choices[0];

        //if no schema, return content
        if (!options.result_schema) {
            return {
                result: choice.message.content as string,
                token_usage: tokenInfo,
                finish_reason: choice.finish_reason, //Uses expected "stop" , "length" format
            }
        }

        //we have a schema: get the content and return after validation
        const data = choice?.message.tool_calls?.[0].function.arguments;
        if (!data) {
            this.logger?.error("[OpenAI] Response is not valid", result);
            throw new Error("Response is not valid: no data");
        }

        return {
            result: data,
            token_usage: tokenInfo,
            finish_reason: choice.finish_reason,
        };
    }

    async requestCompletionStream(prompt: OpenAI.Chat.Completions.ChatCompletionMessageParam[], options: ExecutionOptions): Promise<any> {
        const mapFn = options.result_schema
            ? (chunk: OpenAI.Chat.Completions.ChatCompletionChunk) => {
                return {
                    result: chunk.choices[0]?.delta?.tool_calls?.[0].function?.arguments ?? "",
                    finish_reason: chunk.choices[0]?.finish_reason,         //Uses expected "stop" , "length" format
                    token_usage: {
                        prompt: chunk.usage?.prompt_tokens,
                        result: chunk.usage?.completion_tokens,
                        total: (chunk.usage?.prompt_tokens ?? 0) + (chunk.usage?.completion_tokens ?? 0),
                    }
                } as CompletionChunkObject;
            }
            : (chunk: OpenAI.Chat.Completions.ChatCompletionChunk) => {
                return {
                    result: chunk.choices[0]?.delta.content ?? "",
                    finish_reason: chunk.choices[0]?.finish_reason,
                    token_usage: {
                        prompt: chunk.usage?.prompt_tokens,
                        result: chunk.usage?.completion_tokens,
                        total: (chunk.usage?.prompt_tokens ?? 0) + (chunk.usage?.completion_tokens ?? 0),
                    }
                } as CompletionChunkObject;
            };

        //TODO: OpenAI o1 support requires max_completions_tokens
        const stream = (await this.service.chat.completions.create({
            stream: true,
            stream_options: {include_usage: true},
            model: options.model,
            messages: prompt,
            temperature: options.temperature,
            top_p: options.top_p,
            //top_logprobs: options.top_logprobs,       //Logprobs output currently not supported
            //logprobs: options.top_logprobs ? true : false,
            presence_penalty: options.presence_penalty,
            frequency_penalty: options.frequency_penalty,
            n: 1,
            max_tokens: options.max_tokens,
            tools: options.result_schema
                ? [
                    {
                        function: {
                            name: "format_output",
                            parameters: options.result_schema as any,
                        },
                        type: "function"
                    } as OpenAI.Chat.ChatCompletionTool,
                ]
                : undefined,
            tool_choice: options.result_schema
                ? {
                    type: 'function',
                    function: { name: "format_output" }
                } : undefined,
        })) as Stream<OpenAI.Chat.Completions.ChatCompletionChunk>;

        return asyncMap(stream, mapFn);
    }

    async requestCompletion(prompt: OpenAI.Chat.Completions.ChatCompletionMessageParam[], options: ExecutionOptions): Promise<any> {
        const functions = options.result_schema
            ? [
                {
                    function: {
                        name: "format_output",
                        parameters: options.result_schema as any,
                    },
                    type: 'function'
                } as OpenAI.Chat.ChatCompletionTool,
            ]
            : undefined;

        //TODO: OpenAI o1 support requires max_completions_tokens
        const res = await this.service.chat.completions.create({
            stream: false,
            model: options.model,
            messages: prompt,
            temperature: options.temperature,
            top_p: options.top_p,
            //top_logprobs: options.top_logprobs,       //Logprobs output currently not supported
            //logprobs: options.top_logprobs ? true : false,
            presence_penalty: options.presence_penalty,
            frequency_penalty: options.frequency_penalty,
            n: 1,
            max_tokens: options.max_tokens,
            tools: functions,
            tool_choice: options.result_schema
                ? {
                    type: 'function',
                    function: { name: "format_output" }
                } : undefined,
            // functions: functions,
            // function_call: options.result_schema
            //     ? { name: "format_output" }
            //     : undefined,
        });

        const completion = this.extractDataFromResponse(options, res);
        if (options.include_original_response) {
            completion.original_response = res;
        }
        return completion;
    }

    createTrainingPrompt(options: TrainingPromptOptions): Promise<string> {
        if (options.model.includes("gpt")) {
            return super.createTrainingPrompt(options);
        } else {
            // babbage, davinci not yet implemented
            throw new Error("Unsupported model for training: " + options.model);
        }
    }

    async startTraining(dataset: DataSource, options: TrainingOptions): Promise<TrainingJob> {
        const url = await dataset.getURL();
        const file = await this.service.files.create({
            file: await fetch(url),
            purpose: "fine-tune",
        });

        const job = await this.service.fineTuning.jobs.create({
            training_file: file.id,
            model: options.model,
            hyperparameters: options.params
        })

        return jobInfo(job);
    }

    async cancelTraining(jobId: string): Promise<TrainingJob> {
        const job = await this.service.fineTuning.jobs.cancel(jobId);
        return jobInfo(job);
    }

    async getTrainingJob(jobId: string): Promise<TrainingJob> {
        const job = await this.service.fineTuning.jobs.retrieve(jobId);
        return jobInfo(job);
    }

    // ========= management API =============

    async validateConnection(): Promise<boolean> {
        try {
            await this.service.models.list();
            return true;
        } catch (error) {
            return false;
        }
    }

    listTrainableModels(): Promise<AIModel<string>[]> {
        return this._listModels((m) => supportFineTunning.has(m.id));
    }

    async listModels(): Promise<AIModel[]> {
        return this._listModels();
    }

    async _listModels(filter?: (m: OpenAI.Models.Model) => boolean): Promise<AIModel[]> {
        let result = await this.service.models.list();
        const models = filter ? result.data.filter(filter) : result.data;
        return models.map((m) => ({
            id: m.id,
            name: m.id,
            provider: this.provider,
            owner: m.owned_by,
            type: m.object === "model" ? ModelType.Text : ModelType.Unknown,
            can_stream: true,
            is_multimodal: m.id.includes("gpt-4")
        }));
    }


    async generateEmbeddings({ text, image, model = "text-embedding-3-small" }: EmbeddingsOptions): Promise<EmbeddingsResult> {

        if (image) {
            throw new Error("Image embeddings not supported by OpenAI");
        }

        if (!text) {
            throw new Error("No text provided");
        }

        const res = await this.service.embeddings.create({
            input: text,
            model: model,
        });

        const embeddings = res.data[0].embedding;

        if (!embeddings || embeddings.length === 0) {
            throw new Error("No embedding found");
        }

        return { values: embeddings, model } as EmbeddingsResult;
    }

}


function jobInfo(job: OpenAI.FineTuning.Jobs.FineTuningJob): TrainingJob {
    //validating_files`, `queued`, `running`, `succeeded`, `failed`, or `cancelled`.
    const jobStatus = job.status;
    let status = TrainingJobStatus.running;
    let details: string | undefined;
    if (jobStatus === 'succeeded') {
        status = TrainingJobStatus.succeeded;
    } else if (jobStatus === 'failed') {
        status = TrainingJobStatus.failed;
        details = job.error ? `${job.error.code} - ${job.error.message} ${job.error.param ? " [" + job.error.param + "]" : ""}` : "error";
    } else if (jobStatus === 'cancelled') {
        status = TrainingJobStatus.cancelled;
    } else {
        status = TrainingJobStatus.running;
        details = jobStatus;
    }
    return {
        id: job.id,
        model: job.fine_tuned_model || undefined,
        status,
        details
    }
}