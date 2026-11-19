import { json } from '@sveltejs/kit';
import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import path from 'node:path';

type SampleLocation = { name: string; scope: string; lat: number; lon: number };

// Written by 350Package/export_locations.py
const locations: Record<string, SampleLocation> = JSON.parse(
  readFileSync(
    path.join(process.cwd(), '350Package', 'embeddings', 'random_samples', 'locations.json'),
    'utf8'
  )
);

type Candidate = {
  rank: number;
  image_id: string;
  image_path: string;
  associated_caption: string;
  image_similarity: number;
  caption_similarity: number;
  image_in_radius: boolean;
  caption_in_radius: boolean;
  included_by: string;
};

type WorkerResponse = { id: number; results?: Candidate[]; error?: string };
type PendingSearch = { resolve: (results: Candidate[]) => void; reject: (error: Error) => void };

let worker: ReturnType<typeof spawn> | undefined;
let nextRequestId = 1;
const pending = new Map<number, PendingSearch>();
const stopWorker = (reason: Error) => {
  worker?.kill();
  worker = undefined;
  for (const request of pending.values()) request.reject(reason);
  pending.clear();
};

const startWorker = () => {
  if (worker) return worker;
  const root = process.cwd();
  const python =
    process.env.PYTHON_EXECUTABLE || (process.platform === 'win32' ? 'python' : 'python3');
  const child = spawn(
    python,
    [path.join(root, '350Package', 'search_candidates.py'), '--server'],
    { cwd: path.join(root, '350Package'), windowsHide: true }
  );
  worker = child;
  child.stderr.setEncoding('utf8').on('data', (chunk: string) => {
    // Keep startup/model diagnostics available when troubleshooting the local worker.
    process.stderr.write(`[CLIP worker] ${chunk}`);
  });
  createInterface({ input: child.stdout }).on('line', (line) => {
    try {
      const response = JSON.parse(line) as WorkerResponse;
      const request = pending.get(response.id);
      if (!request) return;
      pending.delete(response.id);
      if (response.error) request.reject(new Error(response.error));
      else request.resolve(response.results ?? []);
    } catch (error) {
      stopWorker(error instanceof Error ? error : new Error('Invalid response from CLIP worker.'));
    }
  });
  child.on('error', (error) => stopWorker(error));
  child.on('close', (code) => {
    if (worker !== child) return;
    stopWorker(new Error(`CLIP worker exited${code === null ? '' : ` with code ${code}`}.`));
  });
  child.stdin?.on('error', (error) => stopWorker(error));
  return child;
};

const runSearch = (query: string, imageRadius: number, captionRadius: number) =>
  new Promise<Candidate[]>((resolve, reject) => {
    const child = startWorker();
    const id = nextRequestId++;
    if (!child.stdin) {
      reject(new Error('Could not open the CLIP worker input stream.'));
      return;
    }
    pending.set(id, { resolve, reject });
    child.stdin.write(
      `${JSON.stringify({ id, query, image_radius: imageRadius, caption_radius: captionRadius })}\n`,
      (error) => {
        if (!error) return;
        pending.delete(id);
        reject(error);
      }
    );
  });

export const POST = async ({ request }) => {
  try {
    const {
      query,
      imageMinimumSimilarity = 0.24,
      captionMinimumSimilarity = 0.76,
      resultLimit = null
    } = await request.json();
    if (typeof query !== 'string' || !query.trim()) {
      return json({ error: 'Enter a search query.' }, { status: 400 });
    }
    if (
      typeof imageMinimumSimilarity !== 'number' ||
      imageMinimumSimilarity < -1 ||
      imageMinimumSimilarity > 1 ||
      typeof captionMinimumSimilarity !== 'number' ||
      captionMinimumSimilarity < -1 ||
      captionMinimumSimilarity > 1 ||
      (resultLimit !== null &&
        (!Number.isInteger(resultLimit) || resultLimit < 1 || resultLimit > 50))
    ) {
      return json(
        { error: 'Similarity thresholds must be between -1 and 1, and result limit must be 1–50 when enabled.' },
        { status: 400 }
      );
    }

    const candidates = await runSearch(
      query.trim(),
      1 - imageMinimumSimilarity,
      1 - captionMinimumSimilarity
    );
    return json({
      items: (resultLimit === null ? candidates : candidates.slice(0, resultLimit)).map((candidate) => ({
        ...candidate,
        id: candidate.image_id,
        imageURL: `/api/dataset/image/${encodeURIComponent(candidate.image_id)}`,
        caption_en: candidate.associated_caption,
        volume: Number(candidate.image_id.match(/Volume_(\d+)/)?.[1] ?? 0),
        page: Number(candidate.image_id.match(/Page_(\d+)/)?.[1] ?? 0),
        location: locations[candidate.image_id] ?? null
      }))
    });
  } catch (error) {
    console.error('CLIP search failed:', error);
    return json(
      { error: 'CLIP search failed. Check the Python and OpenCLIP setup, then try again.' },
      { status: 500 }
    );
  }
};
