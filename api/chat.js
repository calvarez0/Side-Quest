// Vercel Serverless Function - Claude API Proxy
// This keeps your API key secure on the server

export default async function handler(req, res) {
    // Only allow POST
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    // Check for API key
    if (!process.env.ANTHROPIC_API_KEY) {
        return res.status(500).json({ error: 'API key not configured' });
    }

    try {
        const response = await fetch('https://api.anthropic.com/v1/messages', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': process.env.ANTHROPIC_API_KEY,
                'anthropic-version': '2023-06-01'
            },
            body: JSON.stringify(req.body)
        });

        const data = await response.json();

        // Forward the response status and data
        res.status(response.status).json(data);

    } catch (error) {
        console.error('API error:', error);
        res.status(500).json({ error: 'Failed to call Claude API' });
    }
}
