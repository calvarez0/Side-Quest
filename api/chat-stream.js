// Vercel Serverless Function - Claude API Streaming Proxy
// Handles streaming responses for real-time text generation

export const config = {
    runtime: 'edge', // Use edge runtime for streaming
};

export default async function handler(req) {
    if (req.method !== 'POST') {
        return new Response(JSON.stringify({ error: 'Method not allowed' }), {
            status: 405,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
        return new Response(JSON.stringify({ error: 'API key not configured' }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    try {
        const body = await req.json();

        const response = await fetch('https://api.anthropic.com/v1/messages', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': apiKey,
                'anthropic-version': '2023-06-01'
            },
            body: JSON.stringify({
                ...body,
                stream: true
            })
        });

        if (!response.ok) {
            const error = await response.text();
            return new Response(error, {
                status: response.status,
                headers: { 'Content-Type': 'application/json' }
            });
        }

        // Stream the response back
        return new Response(response.body, {
            headers: {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        });

    } catch (error) {
        console.error('Streaming API error:', error);
        return new Response(JSON.stringify({ error: 'Failed to call Claude API' }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}
