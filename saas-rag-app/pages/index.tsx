"use client"

import { useEffect, useRef, useState, type CSSProperties } from 'react';
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
    const [fileValidationError, setFileValidationError] = useState<string>('');
    const [isIngesting, setIsIngesting] = useState<boolean>(false);
    const [fileInputKey, setFileInputKey] = useState<number>(0);
    const [isImageEnabled, setIsImageEnabled] = useState<boolean>(false);
    const [splitPercent, setSplitPercent] = useState<number>(70);
    const splitContainerRef = useRef<HTMLDivElement | null>(null);
    const isDraggingSplitRef = useRef<boolean>(false);

    const maxFileSizeBytes = 5 * 1024 * 1024;
    const allowedExtensions = new Set([
        'md',
        'markdown',
        'txt',
        'pdf',
        'docx',
        'doc',
        'pptx',
        'ppt',
        'xlsx',
        'xls',
        'html',
        'htm',
        'mp3',
        'wav',
        'm4a',
        'flac',
    ]);
    const minSplitPercent = 25;
    const maxSplitPercent = 75;

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
    const splitStyle = {
        ['--left-panel' as string]: `${splitPercent}%`,
        ['--right-panel' as string]: `${100 - splitPercent}%`,
    } as CSSProperties;

    const updateSplitPercent = (clientX: number) => {
        if (!splitContainerRef.current) return;
        const rect = splitContainerRef.current.getBoundingClientRect();
        const rawPercent = ((clientX - rect.left) / rect.width) * 100;
        const clamped = Math.min(maxSplitPercent, Math.max(minSplitPercent, rawPercent));
        setSplitPercent(clamped);
    };

    useEffect(() => {
        const handleMouseMove = (event: MouseEvent) => {
            if (!isDraggingSplitRef.current) return;
            updateSplitPercent(event.clientX);
        };

        const handleMouseUp = () => {
            if (!isDraggingSplitRef.current) return;
            isDraggingSplitRef.current = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        };

        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);

        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, []);

    const handleDividerMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
        event.preventDefault();
        isDraggingSplitRef.current = true;
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        updateSplitPercent(event.clientX);
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
        setFileValidationError('');

        if (!file) {
            setIngestFile(null);
            return;
        }

        const extension = file.name.split('.').pop()?.toLowerCase();
        if (!extension || !allowedExtensions.has(extension)) {
            setFileValidationError('File type not supported.');
            setIngestFile(null);
            setFileInputKey((prev) => prev + 1);
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
        <main className="relative min-h-screen overflow-hidden px-6 pt-20 pb-6 font-sans flex flex-col items-center justify-start gap-5 text-slate-100">
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
            <SignedOut>
                <div className="w-full flex-1 flex flex-col items-center justify-center text-center gap-6">
                    <h1
                        className="text-5xl font-bold tracking-tight text-slate-100 sm:text-6xl"
                        style={{ fontFamily: "'Merriweather', 'Georgia', serif" }}
                    >
                        Personal RAG Agent
                    </h1>
                    <div className="w-full max-w-xl flex items-center justify-center">
                        <SignInButton mode="modal">
                            <button className="inline-flex items-center gap-2 px-8 py-4 rounded-md bg-blue-600 text-lg text-white hover:bg-blue-700 shadow-sm">
                                Try it out for free
                            </button>
                        </SignInButton>
                    </div>
                    <section className="w-full max-w-2xl pt-4 text-center">
                        <p
                            className="text-lg text-slate-300"
                            style={{ fontFamily: "'Comic Sans MS', 'Comic Neue', cursive" }}
                        >
                            Turn any website or document into expert knowledge and chat with it using natural language.
                        </p>

                        <div
                            className="mt-6 text-base text-slate-400"
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
                </div>
            </SignedOut>

            <SignedIn>
                <h1
                    className="text-3xl font-bold text-center text-slate-100"
                    style={{ fontFamily: "'Merriweather', 'Georgia', serif" }}
                >
                    Personal RAG Agent
                </h1>
                <div className="w-full max-w-none">
                    <div
                        ref={splitContainerRef}
                        style={splitStyle}
                        className="flex w-full min-h-[60vh] flex-col gap-6 lg:flex-row lg:gap-0 lg:rounded-2xl lg:border lg:border-white/10 lg:bg-white/5 lg:backdrop-blur"
                    >
                        <div className="w-full lg:flex-none lg:basis-[var(--left-panel)] lg:px-6 lg:py-6">
                            <h2
                                className="mb-4 text-xl font-semibold text-slate-100"
                                style={{ fontFamily: "'Merriweather', 'Georgia', serif" }}
                            >
                                Start conversation with your RAG agent
                            </h2>
            <form
                onSubmit={handleSubmit}
                className="w-full p-6 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg shadow-sm space-y-4"
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
                        </div>
                        <div
                            onMouseDown={handleDividerMouseDown}
                            role="separator"
                            aria-orientation="vertical"
                            aria-valuenow={Math.round(splitPercent)}
                            aria-valuemin={minSplitPercent}
                            aria-valuemax={maxSplitPercent}
                            className="group hidden w-4 cursor-col-resize items-stretch justify-center lg:flex"
                            title="Drag to resize"
                        >
                            <div className="my-6 w-px rounded-full bg-white/20 transition-colors group-hover:bg-white/60" />
                        </div>
                        <div className="w-full lg:flex-none lg:basis-[var(--right-panel)] lg:px-6 lg:py-6">
                            <h2
                                className="mb-4 text-xl font-semibold text-slate-100"
                                style={{ fontFamily: "'Merriweather', 'Georgia', serif" }}
                            >
                                Create Knowledge Base
                            </h2>
            <form
                onSubmit={handleIngestionSubmit}
                className="w-full p-6 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg shadow-sm space-y-4"
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
                    <div className="relative">
                        <input
                            key={fileInputKey}
                            type="file"
                            onChange={handleFileChange}
                            accept=".md,.markdown,.txt,.pdf,.docx,.doc,.pptx,.ppt,.xlsx,.xls,.html,.htm,.mp3,.wav,.m4a,.flac"
                            className="w-full rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-sm text-slate-700 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500/30 file:mr-4 file:rounded-l-md file:rounded-r-none file:border-0 file:border-r file:border-slate-300 dark:file:border-slate-600 file:bg-slate-200 dark:file:bg-slate-700 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-slate-800 dark:file:text-slate-100"
                        />
                        {fileValidationError && (
                            <div className="mt-2 inline-flex items-center gap-2 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs font-medium text-red-700 shadow-sm dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-200">
                                {fileValidationError}
                            </div>
                        )}
                    </div>
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
                        Enable image recognition for pdf files
                    </label>
                )}
            </form>
                        </div>
                    </div>
                </div>
                <section className="w-full max-w-2xl pt-2 text-center">
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
            </SignedIn>
        </main>
    );
}
