import { dirname, join } from 'path';
import { readFileSync } from 'fs';


const dataDir = join(dirname(new URL(import.meta.url).pathname), 'data');
const dataFile = (file: string) => join(dataDir, file);
const readDataFile = (file: string, enc: BufferEncoding = 'utf-8') => {
    return readFileSync(dataFile(file), enc);
}
export { dataDir, dataFile, readDataFile };
