"use client"

import { useState, type CSSProperties } from 'react';
import { SignInButton, SignedIn, SignedOut, UserButton, useAuth } from '@clerk/nextjs';
import ReactMarkdown from 'react-markdown';
import remarkGfm from "remark-gfm";

function Markdown({ children }: { children: string }) {
    return (
        <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
                a: ({ ...props }) => (
                    <a
                        {...props}
                        className="text-blue-600 dark:text-blue-400 underline underline-offset-2"
                        target={props.href?.startsWith('#') ? undefined : "_blank"}
                        rel={props.href?.startsWith('#') ? undefined : "noreferrer"}
                    />
                ),
                p: ({ ...props }) => <p {...props} className="whitespace-pre-wrap" />,
                ul: ({ ...props }) => <ul {...props} className="list-disc pl-6 space-y-1" />,
                ol: ({ ...props }) => <ol {...props} className="list-decimal pl-6 space-y-1" />,
                li: ({ ...props }) => <li {...props} />,
                code: ({ children, className, ...props }) => (
                    <code
                        {...props}
                        className={[
                            "rounded bg-gray-100 dark:bg-gray-900 px-1 py-0.5",
                            className ?? "",
                        ].join(" ")}
                    >
                        {children}
                    </code>
                ),
                pre: ({ ...props }) => (
                    <pre
                        {...props}
                        className="overflow-x-auto rounded bg-gray-100 dark:bg-gray-900 p-3"
                    />
                ),
            }}
        >
            {children}
        </ReactMarkdown>
    );
}

export default function Home() {
    const { getToken } = useAuth();
    const [idea, setIdea] = useState<string>('');
    const [inputText, setInputText] = useState<string>('');
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [ingestUrl, setIngestUrl] = useState<string>('');
    const [ingestFile, setIngestFile] = useState<File | null>(null);
    const [ingestMode, setIngestMode] = useState<'url' | 'file'>('url');
    const [ingestion_response, setIngestionResponse] = useState<string>('');
    const [ingestionError, setIngestionError] = useState<string>('');
    const [isIngesting, setIsIngesting] = useState<boolean>(false);
    const [fileInputKey, setFileInputKey] = useState<number>(0);
    const [isImageEnabled, setIsImageEnabled] = useState<boolean>(false);

    const maxFileSizeBytes = 5 * 1024 * 1024;

    const wallpaperDark: CSSProperties = {
        backgroundColor: '#0f172a',
        backgroundImage: [
            'linear-gradient(135deg, #0f172a 0%, #1e293b 42%, #0b3a4a 100%)',
            'radial-gradient(1200px 720px at 18% -8%, rgba(56,189,248,0.22), transparent 62%)',
            'radial-gradient(1000px 700px at 110% 18%, rgba(20,184,166,0.22), transparent 58%)',
            'radial-gradient(900px 600px at 55% 110%, rgba(148,163,184,0.18), transparent 60%)',
        ].join(', '),
        backgroundSize: 'cover',
    };

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (isLoading) return;

        setIdea('');
        setIsLoading(true);

        try {
            const jwt = await getToken();
            if (!jwt) {
                setIdea('Authentication required');
                setIsLoading(false);
                return;
            }

            const res = await fetch('/api', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${jwt}`,
                },
                body: JSON.stringify({ text: inputText }),
            });

            const text = await res.text();
            if (!res.ok) throw new Error(text || `Request failed (${res.status})`);

            setIdea(text);
        } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            setIdea('Error: ' + message);
        } finally {
            setIsLoading(false);
        }
    };

    const handleUrlIngestion = async (event?: React.FormEvent<HTMLFormElement> | React.MouseEvent<HTMLButtonElement>) => {
        event?.preventDefault();
        if (isIngesting) return;

        setIngestionResponse('');
        setIngestionError('');
        setIsIngesting(true);

        try {
            const jwt = await getToken();
            if (!jwt) {
                setIngestionError('Authentication required');
                setIsIngesting(false);
                return;
            }

            const res = await fetch('/ingest', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${jwt}`,
                },
                body: JSON.stringify({ url: ingestUrl }),
            });

            const text = await res.text();
            if (!res.ok) throw new Error(text || `Request failed (${res.status})`);

            setIngestionResponse(text);
        } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            setIngestionResponse('Error: ' + message);
        } finally {
            setIsIngesting(false);
        }
    };

    const handleFileIngestion = async (event?: React.FormEvent<HTMLFormElement> | React.MouseEvent<HTMLButtonElement>) => {
        event?.preventDefault();
        if (isIngesting) return;

        if (!ingestFile) {
            setIngestionError('Please select a file to ingest.');
            return;
        }

        if (ingestFile.size > maxFileSizeBytes) {
            setIngestionError('File size cannot exceed 5 MB.');
            return;
        }

        setIngestionResponse('');
        setIngestionError('');
        setIsIngesting(true);

        try {
            const jwt = await getToken();
            if (!jwt) {
                setIngestionError('Authentication required');
                setIsIngesting(false);
                return;
            }

            const formData = new FormData();
            formData.append('file', ingestFile);
            formData.append('isImageEnabled', String(isImageEnabled));

            const res = await fetch('/ingest-file', {
                method: 'POST',
                headers: {
                    Authorization: `Bearer ${jwt}`,
                },
                body: formData,
            });

            if (res.status !== 202) {
                const text = await res.text();
                if (!res.ok) throw new Error(text || `Request failed (${res.status})`);
                setIngestionResponse(text);
                return;
            }

            const data = (await res.json()) as { job_id?: string };
            const jobId = data.job_id;
            if (!jobId) throw new Error('Backend did not return a job_id.');

            setIngestionResponse(`Ingestion started (job_id: ${jobId}).`);

            const pollEveryMs = 1000;
            const timeoutMs = 10 * 60 * 1000;
            const startedAt = Date.now();

            while (true) {
                if (Date.now() - startedAt > timeoutMs) {
                    throw new Error('Ingestion is taking too long. Please check backend logs.');
                }

                // Wait pollEveryMs milliseconds, then continue.
                await new Promise((resolve) => setTimeout(resolve, pollEveryMs));

                const statusRes = await fetch(`/ingest-file/status/${jobId}`);
                if (!statusRes.ok) {
                    const text = await statusRes.text();
                    throw new Error(text || `Status request failed (${statusRes.status})`);
                }
                const statusData = (await statusRes.json()) as { status?: string; error?: string };
                const status = statusData.status ?? 'unknown';

                if (status === 'succeeded') {
                    setIngestionResponse('Successfully ingested the document.');
                    break;
                }

                if (status === 'failed') {
                    throw new Error(statusData.error || 'Ingestion failed.');
                }

                setIngestionResponse(`Ingestion status: ${status}...`);
            }
        } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            setIngestionResponse('Error: ' + message);
        } finally {
            setIsIngesting(false);
        }
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0] ?? null;
        setIngestionError('');

        if (!file) {
            setIngestFile(null);
            return;
        }

        if (file.size > maxFileSizeBytes) {
            setIngestionError('File size cannot exceed 5 MB.');
            setIngestFile(null);
            setFileInputKey((prev) => prev + 1);
            return;
        }

        setIngestFile(file);
    };

    const handleIngestionSubmit = (event: React.FormEvent<HTMLFormElement>) => {
        if (ingestMode === 'url') {
            void handleUrlIngestion(event);
            return;
        }

        void handleFileIngestion(event);
    };

    const handleIngestButtonClick = (
        mode: 'url' | 'file',
        handler: (event?: React.MouseEvent<HTMLButtonElement>) => Promise<void>
    ) => async (event: React.MouseEvent<HTMLButtonElement>) => {
        if (ingestMode !== mode) {
            setIngestMode(mode);
            return;
        }

        await handler(event);
    };

    return (
        <main className="relative min-h-screen overflow-hidden px-6 pt-24 pb-10 font-sans flex flex-col items-center justify-center gap-6 text-slate-100">
            <div aria-hidden="true" className="pointer-events-none absolute inset-0 -z-10">
                <div className="absolute inset-0" style={wallpaperDark} />
            </div>
            <nav className="fixed top-0 left-0 right-0 z-20 bg-gradient-to-r from-emerald-800 via-green-700 to-emerald-600 shadow-md">
                <div className="w-full flex items-center justify-end px-6 py-3">
                    <SignedOut>
                        <SignInButton mode="modal">
                            <button className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-white/95 text-emerald-800 hover:bg-white shadow-sm border border-white/70">
                                Sign In
                            </button>
                        </SignInButton>
                    </SignedOut>
                    <SignedIn>
                        <UserButton />
                    </SignedIn>
                </div>
            </nav>
            <h1
                className="text-3xl font-bold text-center text-slate-100"
                style={{ fontFamily: "'Merriweather', 'Georgia', serif" }}
            >
                Personal RAG Agent
            </h1>
            <SignedOut>
                <div className="w-full max-w-2xl flex items-center justify-center">
                    <SignInButton mode="modal">
                        <button className="inline-flex items-center gap-2 px-6 py-3 rounded-md bg-blue-600 text-white hover:bg-blue-700 shadow-sm">
                            Try it out for free
                        </button>
                    </SignInButton>
                </div>
            </SignedOut>

            <SignedIn>
            <form
                onSubmit={handleSubmit}
                className="w-full max-w-2xl p-6 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg shadow-sm space-y-4"
                aria-busy={isLoading}
            >
                <input
                    type="text"
                    value={inputText}
                    onChange={(event) => setInputText(event.target.value)}
                    placeholder="Ask any question about your documents..."
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-transparent text-gray-900 dark:text-gray-100"
                />
                <button
                    type="submit"
                    disabled={isLoading}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-60 disabled:cursor-not-allowed"
                >
                    {isLoading && (
                        <span
                            className="h-4 w-4 rounded-full border-2 border-white/40 border-t-white animate-spin"
                            aria-hidden="true"
                        />
                    )}
                    {isLoading ? 'Generating…' : 'Generate'}
                </button>
                {idea && (
                    <div className="text-gray-900 dark:text-gray-100">
                        <Markdown>{idea}</Markdown>
                    </div>
                )}
            </form>

            <form
                onSubmit={handleIngestionSubmit}
                className="w-full max-w-2xl p-6 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg shadow-sm space-y-4"
                aria-busy={isIngesting}
            >
                {ingestMode === 'url' ? (
                    <input
                        type="url"
                        value={ingestUrl}
                        onChange={(event) => setIngestUrl(event.target.value)}
                        placeholder="Enter website url to ingest"
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-transparent text-gray-900 dark:text-gray-100"
                    />
                ) : (
                            <input
                                key={fileInputKey}
                                type="file"
                                onChange={handleFileChange}
                                className="w-full rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-sm text-slate-700 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500/30 file:mr-4 file:rounded-l-md file:rounded-r-none file:border-0 file:border-r file:border-slate-300 dark:file:border-slate-600 file:bg-slate-200 dark:file:bg-slate-700 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-slate-800 dark:file:text-slate-100"
                            />
                )}
                <div className="flex flex-wrap items-center gap-3">
                    <button
                        type="button"
                        disabled={isIngesting}
                        onClick={handleIngestButtonClick('url', handleUrlIngestion)}
                        className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-green-600 text-white hover:bg-green-700 disabled:opacity-60 disabled:cursor-not-allowed"
                    >
                        {isIngesting && ingestMode === 'url' && (
                            <span
                                className="h-4 w-4 rounded-full border-2 border-white/40 border-t-white animate-spin"
                                aria-hidden="true"
                            />
                        )}
                        {isIngesting && ingestMode === 'url' ? 'Ingesting data...' : 'Ingest url'}
                    </button>
                    <button
                        type="button"
                        disabled={isIngesting}
                        onClick={handleIngestButtonClick('file', handleFileIngestion)}
                        className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-emerald-600 text-white hover:bg-emerald-700 disabled:opacity-60 disabled:cursor-not-allowed"
                    >
                        {isIngesting && ingestMode === 'file' && (
                            <span
                                className="h-4 w-4 rounded-full border-2 border-white/40 border-t-white animate-spin"
                                aria-hidden="true"
                            />
                        )}
                        {isIngesting && ingestMode === 'file' ? 'Ingesting data...' : 'Ingest File'}
                    </button>
                </div>
                {ingestionError && (
                    <div className="text-sm text-red-600 dark:text-red-400">
                        {ingestionError}
                    </div>
                )}
                {ingestion_response && (
                    <div className="text-gray-900 dark:text-gray-100">
                        <Markdown>{ingestion_response}</Markdown>
                    </div>
                )}
                {ingestMode === 'file' && (
                    <label
                        className="mt-2 flex items-center gap-2 text-sm text-gray-900 dark:text-gray-100"
                        style={{ fontFamily: "'Merriweather', 'Georgia', serif" }}
                    >
                        <input
                            type="checkbox"
                            checked={isImageEnabled}
                            onChange={(event) => setIsImageEnabled(event.target.checked)}
                            className="h-4 w-4 rounded border-gray-300 dark:border-gray-600"
                        />
                        Enable image extraction for pdf files (This will increase the ingestion time)
                    </label>
                )}
            </form>
            </SignedIn>

            <section className="w-full max-w-2xl text-center">
                <p
                    className="text-base text-slate-300"
                    style={{ fontFamily: "'Comic Sans MS', 'Comic Neue', cursive" }}
                >
                    Turn any website or document into expert knowledge and chat with it using natural language.
                </p>

                <div
                    className="mt-6 text-xs text-slate-400"
                    style={{ fontFamily: "'Comic Sans MS', 'Comic Neue', cursive" }}
                >
                    <span>
                        &copy; {new Date().getFullYear()} Personal RAG Agent
                    </span>
                    <span className="ml-2">
                        All rights reserved.
                    </span>
                </div>
            </section>
        </main>
    );
}
