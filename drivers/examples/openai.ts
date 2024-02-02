import { AIModel, PromptRole, PromptSegment } from "@llumiverse/core";
import { OpenAIDriver } from "@llumiverse/drivers";

async function main() {

    const vertexai = new OpenAIDriver({
        apiKey: "YOUR_API_KEY_HERE",
        logger: false
    });

    // list models
    const models: AIModel[] = await vertexai.listModels();

    console.log('# Available OpenAI Models:');
    for (const model of models) {
        console.log(`${model.name} [${model.id}]`);
    }

    // execute a model (blocking)
    const prompt: PromptSegment[] = [
        {
            role: PromptRole.user,
            content: 'Write please a short story about Paris in winter in no more than 512 characters.'
        }
    ]

    console.log('\n# Executing model text-bison with prompt: ', prompt);
    const response = await vertexai.execute(prompt, {
        model: 'gpt-3.5-turbo',
        temperature: 0.6,
        max_tokens: 1024
    });

    console.log('\n# LLM response:', response.result)
    console.log('# Response took', response.execution_time, 'ms')
    console.log('# Token usage:', response.token_usage);

    // execute a model in streaming mode 
    console.log('\n# Executing model text-bison in streaming mode with prompt: ', prompt);
    const stream = await vertexai.stream(prompt, {
        model: 'gpt-3.5-turbo',
        temperature: 0.6,
        max_tokens: 1024
    });

    // show the streaming response as it comes
    for await (const chunk of stream) {
        process.stdout.write(chunk);
    }

    // get the recomposed response from the stream chunks
    const streamingResponse = stream.completion!;

    console.log('\n# LLM response:', streamingResponse.result)
    console.log('# Response took', streamingResponse.execution_time, 'ms')
    console.log('# Token usage:', streamingResponse.token_usage);

}

main().catch(console.error);
