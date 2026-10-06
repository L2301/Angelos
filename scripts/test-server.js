import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createLab } from '../src/server.js';

const dataDirectory = await mkdtemp(join(tmpdir(), 'angelos-browser-'));
const { server } = await createLab({ dataDirectory });
server.listen(4311, '127.0.0.1', () => process.stdout.write('Isolated browser test server ready on 4311\n'));
