// ============================================
// SIDE QUEST - A Curiosity Journey
// Using Claude Haiku 4.5 + CPPN-style Curiosity Network
// Inspired by Jeff Clune / Ken Stanley Open-Endedness
// ============================================

const CONFIG = {
    model: 'claude-haiku-4-5-20251001',
    numQuestions: 6,
    mutationRate: 0.3,
    noveltyThreshold: 0.3,
    responseSentences: 3
};

const SEED_ARCHIVE_PAGE_SIZE = 6;

const SEED_ARCHIVE_FALLBACK = [
    'Why are pineapples spiky?',
    'How do I know I am really me?',
    'Where does yesterday go?',
    'Who was the coolest designer?',
    'Which civilization lasted the longest?',
    'Why does the universe feel personal sometimes?',
    'Why do we trust stories more than statistics?',
    'How does memory decide what to keep?',
    'Why do old songs feel like time travel?',
    'Could cities be designed like forests?',
    'Why do maps change how we think?',
    'How do rituals shape identity?',
    'Can boredom be a creative superpower?',
    'Why are some ideas timeless?',
    'How do languages change what we notice?',
    'Could architecture improve mental health?',
    'Why do symbols feel powerful?',
    'How does myth survive modern life?',
    'Can AI make us more human?',
    'Why does beauty feel objective sometimes?',
    'How do people invent new genres?',
    'Can a question be better than an answer?',
    'Why do we crave mystery?',
    'How does curiosity spread between people?',
    'Can technology feel sacred?',
    'Why are thresholds and doorways so symbolic?',
    'How do children ask better questions than adults?',
    'Can design reduce loneliness?',
    'Why does nostalgia feel warm and painful at once?',
    'How do communities keep knowledge alive?',
    'What makes a place feel alive?',
    'Can ordinary objects become meaningful art?',
    'Why do we search for meaning in patterns?',
    'How does humor make difficult truths bearable?',
    'Can food be a language?',
    'Why do ideas return in new forms?',
    'How does music create belonging?',
    'Can future cities be emotionally intelligent?',
    'Why do we romanticize the unknown?',
    'How can we design for wonder, not just efficiency?'
];

const PROMPTBREEDER_TASK_DESCRIPTION = 'Generate follow-up curiosity questions for Side Quest. Questions should be short, inviting, and exploratory.';
const PROMPTBREEDER_HYPER_MUTATION_PROMPT = 'Please summarize and improve the following instruction:';

const PROMPTBREEDER_SEED_TASK_PROMPTS = [
    'Generate follow-up questions that deepen the current idea while staying clear and human.',
    'Generate six concise follow-up questions that feel curious, vivid, and easy to answer.',
    'Generate follow-up questions that branch to fresh angles without losing connection to the current answer.',
    'Generate follow-up questions that balance wonder, clarity, and practical relevance.',
    'Generate follow-up questions that invite reflection and discovery rather than debate.'
];

const PROMPTBREEDER_THINKING_STYLES = [
    'Let\'s think step by step.',
    'Reframe the problem from a contrasting perspective.',
    'Seek one unexpected but relevant angle.',
    'Prioritize clarity over cleverness.',
    'Start concrete, then open toward abstract insight.',
    'Use analogies to spark new lines of inquiry.',
    'Find the hidden assumption and question it.',
    'Generate ideas that feel both surprising and useful.'
];

const PROMPTBREEDER_MUTATION_PROMPTS = [
    'Rewrite this instruction to be more specific and practical while keeping it concise.',
    'Say that instruction again in another way. Do not copy phrases verbatim.',
    'Make the instruction more imaginative but still easy to follow.',
    'Mutate the instruction with an unexpected twist that still serves the task.',
    'Improve the instruction for clarity, warmth, and actionability.',
    'Rewrite the instruction so it encourages broader exploration.',
    'Rewrite the instruction so it encourages deeper reasoning.',
    'Produce a variant that sounds natural to a curious guide.'
];

const PROMPTBREEDER_OPERATORS = [
    'direct_first_order',
    'direct_zero_order',
    'eda_population',
    'eda_ranked',
    'eda_lineage',
    'hyper_zero_order',
    'hyper_first_order',
    'lamarckian',
    'crossover_context_shuffle'
];

const PROMPTBREEDER_HISTORY_LIMIT = 24;

// ============================================
// MARKDOWN PARSING - Convert Claude's markdown to plain text
// ============================================

function parseMarkdown(text) {
    if (!text) return text;

    return text
        // Bold: **text** or __text__
        .replace(/\*\*([^*]+)\*\*/g, '$1')
        .replace(/__([^_]+)__/g, '$1')
        // Italic: *text* or _text_
        .replace(/\*([^*]+)\*/g, '$1')
        .replace(/(?<!\w)_([^_]+)_(?!\w)/g, '$1')
        // Code: `text`
        .replace(/`([^`]+)`/g, '$1')
        // Strikethrough: ~~text~~
        .replace(/~~([^~]+)~~/g, '$1')
        // Headers: # text
        .replace(/^#{1,6}\s+/gm, '')
        // Links: [text](url) -> text
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
        // Bullet points: - text or * text
        .replace(/^[\-\*]\s+/gm, '')
        // Clean up extra whitespace
        .replace(/\n{3,}/g, '\n\n')
        .trim();
}

// ============================================
// CURIOSITYNET - Neural Network for Curiosity
// A CPPN-style network that maps semantic features
// to curiosity dimensions, evolving via NEAT-like ops
// ============================================

class CuriosityNet {
    constructor() {
        this.innovationNumber = 0;
        this.mutationCount = 0;
        this.nodes = new Map();
        this.connections = [];
        this.initializeNetwork();
    }

    initializeNetwork() {
        // Input nodes (IDs 0-5)
        const inputLabels = ['wordFreq', 'qType', 'abstract', 'novelty', 'depth', 'predErr'];
        for (let i = 0; i < 6; i++) {
            this.nodes.set(i, {
                id: i,
                type: 'input',
                label: inputLabels[i],
                activation: 'linear',
                value: 0
            });
        }

        // Output nodes (IDs 6-11)
        const outputLabels = ['depthDrive', 'breadthDrive', 'abstractDrive', 'mechanismDrive', 'exploreBonus', 'qLength'];
        for (let i = 6; i < 12; i++) {
            this.nodes.set(i, {
                id: i,
                type: 'output',
                label: outputLabels[i - 6],
                activation: 'tanh',
                value: 0
            });
        }

        // Initial connections
        this.addConnection(3, 10, 0.8);  // novelty -> exploreBonus
        this.addConnection(5, 6, 0.5);   // predErr -> depthDrive
        this.addConnection(5, 10, 0.7);  // predErr -> exploreBonus
        this.addConnection(4, 6, 0.3);   // depth -> depthDrive
        this.addConnection(4, 7, -0.3);  // depth -> breadthDrive
        this.addConnection(2, 8, 0.6);   // abstract -> abstractDrive
        this.addConnection(1, 9, 0.5);   // qType -> mechanismDrive
        this.addConnection(4, 11, 0.2);  // depth -> qLength
        this.addConnection(2, 11, -0.3); // abstract -> qLength
    }

    addConnection(fromId, toId, weight) {
        this.connections.push({
            innovation: this.innovationNumber++,
            from: fromId,
            to: toId,
            weight: weight,
            enabled: true
        });
    }

    static activations = {
        linear: x => x,
        tanh: x => Math.tanh(x),
        sigmoid: x => 1 / (1 + Math.exp(-x)),
        relu: x => Math.max(0, x),
        sin: x => Math.sin(x * Math.PI),
        gaussian: x => Math.exp(-x * x * 2),
        abs: x => Math.abs(x),
        step: x => x > 0 ? 1 : 0,
        softplus: x => Math.log(1 + Math.exp(x)),
        curiosity: x => Math.tanh(x) * (1 + 0.1 * Math.sin(x * 3))
    };

    forward(inputs) {
        this.nodes.get(0).value = inputs.wordFreq || 0;
        this.nodes.get(1).value = inputs.qType || 0;
        this.nodes.get(2).value = inputs.abstract || 0;
        this.nodes.get(3).value = inputs.novelty || 0;
        this.nodes.get(4).value = inputs.depth || 0;
        this.nodes.get(5).value = inputs.predErr || 0;

        for (const [id, node] of this.nodes) {
            if (node.type !== 'input') {
                node.value = 0;
            }
        }

        const order = this.topologicalSort();

        for (const nodeId of order) {
            const node = this.nodes.get(nodeId);
            if (node.type === 'input') continue;

            let sum = 0;
            for (const conn of this.connections) {
                if (conn.to === nodeId && conn.enabled) {
                    const fromNode = this.nodes.get(conn.from);
                    sum += fromNode.value * conn.weight;
                }
            }

            const activationFn = CuriosityNet.activations[node.activation] || CuriosityNet.activations.tanh;
            node.value = activationFn(sum);
        }

        return {
            depthDrive: this.nodes.get(6).value,
            breadthDrive: this.nodes.get(7).value,
            abstractDrive: this.nodes.get(8).value,
            mechanismDrive: this.nodes.get(9).value,
            explorationBonus: this.nodes.get(10).value,
            questionLength: this.nodes.get(11).value
        };
    }

    topologicalSort() {
        const visited = new Set();
        const order = [];

        const visit = (nodeId) => {
            if (visited.has(nodeId)) return;
            visited.add(nodeId);

            for (const conn of this.connections) {
                if (conn.to === nodeId && conn.enabled) {
                    visit(conn.from);
                }
            }

            order.push(nodeId);
        };

        for (const [id] of this.nodes) {
            visit(id);
        }

        return order;
    }

    mutate() {
        this.mutationCount++;
        const r = Math.random();

        if (r < 0.6) {
            this.mutateWeights();
        } else if (r < 0.8) {
            this.mutateAddConnection();
        } else if (r < 0.95) {
            this.mutateAddNode();
        } else {
            this.mutateActivation();
        }
    }

    mutateWeights() {
        for (const conn of this.connections) {
            if (Math.random() < 0.3) {
                conn.weight += (Math.random() - 0.5) * 0.4;
                conn.weight = Math.max(-2, Math.min(2, conn.weight));
            }
            if (Math.random() < 0.05) {
                conn.weight = (Math.random() - 0.5) * 2;
            }
        }
    }

    mutateAddConnection() {
        const nodeIds = Array.from(this.nodes.keys());
        const inputIds = nodeIds.filter(id => this.nodes.get(id).type === 'input');
        const outputIds = nodeIds.filter(id => this.nodes.get(id).type !== 'input');

        for (let attempt = 0; attempt < 10; attempt++) {
            const fromId = inputIds[Math.floor(Math.random() * inputIds.length)];
            const toId = outputIds[Math.floor(Math.random() * outputIds.length)];

            const exists = this.connections.some(c => c.from === fromId && c.to === toId);
            if (!exists && fromId !== toId) {
                this.addConnection(fromId, toId, (Math.random() - 0.5) * 2);
                break;
            }
        }
    }

    mutateAddNode() {
        if (this.connections.length === 0) return;

        const enabledConns = this.connections.filter(c => c.enabled);
        if (enabledConns.length === 0) return;

        const conn = enabledConns[Math.floor(Math.random() * enabledConns.length)];
        conn.enabled = false;

        const newId = Math.max(...Array.from(this.nodes.keys())) + 1;
        const activations = ['tanh', 'sigmoid', 'relu', 'sin', 'gaussian', 'curiosity'];

        this.nodes.set(newId, {
            id: newId,
            type: 'hidden',
            label: `h${newId}`,
            activation: activations[Math.floor(Math.random() * activations.length)],
            value: 0
        });

        this.addConnection(conn.from, newId, 1.0);
        this.addConnection(newId, conn.to, conn.weight);
    }

    mutateActivation() {
        const hiddenNodes = Array.from(this.nodes.values()).filter(n => n.type === 'hidden');
        if (hiddenNodes.length === 0) return;

        const node = hiddenNodes[Math.floor(Math.random() * hiddenNodes.length)];
        const activations = ['tanh', 'sigmoid', 'relu', 'sin', 'gaussian', 'curiosity', 'softplus', 'abs'];
        node.activation = activations[Math.floor(Math.random() * activations.length)];
    }

    toJSON() {
        return {
            nodes: Array.from(this.nodes.entries()),
            connections: this.connections,
            innovationNumber: this.innovationNumber,
            mutationCount: this.mutationCount
        };
    }

    static fromJSON(json) {
        const net = new CuriosityNet();
        net.nodes = new Map(json.nodes);
        net.connections = json.connections;
        net.innovationNumber = json.innovationNumber;
        net.mutationCount = json.mutationCount;
        return net;
    }
}

// ============================================
// NOVELTY ARCHIVE - Behavioral Characterization
// ============================================

class NoveltyArchive {
    constructor(maxSize = 100) {
        this.archive = [];
        this.maxSize = maxSize;
        this.frontiersFound = 0;
    }

    characterize(question, answer, context) {
        const qWords = question.toLowerCase().split(/\s+/);
        const aWords = answer.toLowerCase().split(/\s+/);

        return {
            whyRatio: this.countPatterns(question, ['why', 'reason', 'cause']) / qWords.length,
            howRatio: this.countPatterns(question, ['how', 'mechanism', 'process', 'work']) / qWords.length,
            whatRatio: this.countPatterns(question, ['what', 'define', 'explain']) / qWords.length,
            abstractness: this.countPatterns(question, ['concept', 'theory', 'principle', 'nature', 'meaning', 'essence']) / qWords.length,
            concreteness: this.countPatterns(question, ['example', 'specific', 'case', 'instance', 'practical', 'real']) / qWords.length,
            answerComplexity: aWords.length / 100,
            technicalDensity: this.countPatterns(answer, ['algorithm', 'system', 'structure', 'function', 'process', 'method']) / Math.max(aWords.length, 1),
            semanticHash: this.simpleHash(question + answer)
        };
    }

    countPatterns(text, patterns) {
        const lower = text.toLowerCase();
        return patterns.reduce((count, p) => count + (lower.includes(p) ? 1 : 0), 0);
    }

    simpleHash(text) {
        const hash = new Array(8).fill(0);
        const words = text.toLowerCase().split(/\s+/);

        for (let i = 0; i < words.length; i++) {
            const word = words[i];
            for (let j = 0; j < word.length; j++) {
                const idx = (word.charCodeAt(j) * (i + 1)) % 8;
                hash[idx] += 1 / words.length;
            }
        }

        const mag = Math.sqrt(hash.reduce((s, v) => s + v * v, 0)) || 1;
        return hash.map(v => v / mag);
    }

    calculateNovelty(behavior, k = 5) {
        if (this.archive.length === 0) return 1.0;

        const distances = this.archive.map(archived =>
            this.behaviorDistance(behavior, archived.behavior)
        );

        distances.sort((a, b) => a - b);
        const kNearest = distances.slice(0, Math.min(k, distances.length));

        return kNearest.reduce((s, d) => s + d, 0) / kNearest.length;
    }

    behaviorDistance(b1, b2) {
        let dist = 0;

        dist += Math.abs(b1.whyRatio - b2.whyRatio);
        dist += Math.abs(b1.howRatio - b2.howRatio);
        dist += Math.abs(b1.whatRatio - b2.whatRatio);
        dist += Math.abs(b1.abstractness - b2.abstractness);
        dist += Math.abs(b1.concreteness - b2.concreteness);
        dist += Math.abs(b1.answerComplexity - b2.answerComplexity);
        dist += Math.abs(b1.technicalDensity - b2.technicalDensity);

        for (let i = 0; i < 8; i++) {
            dist += Math.abs(b1.semanticHash[i] - b2.semanticHash[i]);
        }

        return dist / 15;
    }

    maybeAdd(question, answer, context, threshold = 0.3) {
        const behavior = this.characterize(question, answer, context);
        const novelty = this.calculateNovelty(behavior);

        if (novelty > threshold || this.archive.length < 10) {
            this.archive.push({
                question,
                answer,
                behavior,
                novelty,
                timestamp: Date.now()
            });

            if (novelty > 0.7) {
                this.frontiersFound++;
            }

            if (this.archive.length > this.maxSize) {
                this.archive.sort((a, b) => b.novelty - a.novelty);
                this.archive = this.archive.slice(0, this.maxSize);
            }

            return { added: true, novelty };
        }

        return { added: false, novelty };
    }

    getVisualizationData() {
        return this.archive.map(item => ({
            x: item.behavior.abstractness - item.behavior.concreteness,
            y: item.behavior.whyRatio - item.behavior.howRatio,
            novelty: item.novelty,
            question: item.question.substring(0, 50)
        }));
    }
}

// ============================================
// PREDICTION MODEL - For Intrinsic Motivation
// ============================================

class PredictionModel {
    constructor() {
        this.weights = {
            complexityWeights: new Array(6).fill(0).map(() => Math.random() * 0.2 - 0.1),
            technicalWeights: new Array(6).fill(0).map(() => Math.random() * 0.2 - 0.1)
        };
        this.learningRate = 0.1;
        this.lastPrediction = null;
        this.lastError = 0;
    }

    extractFeatures(question) {
        const words = question.toLowerCase().split(/\s+/);
        return [
            words.length / 20,
            this.countPatterns(question, ['why', 'how', 'what']) / 3,
            this.countPatterns(question, ['theory', 'concept', 'abstract']) / 3,
            this.countPatterns(question, ['example', 'specific', 'practical']) / 3,
            this.countPatterns(question, ['?']) > 0 ? 1 : 0,
            Math.random() * 0.1
        ];
    }

    countPatterns(text, patterns) {
        const lower = text.toLowerCase();
        return patterns.reduce((count, p) => count + (lower.includes(p) ? 1 : 0), 0);
    }

    predict(question) {
        const features = this.extractFeatures(question);

        const complexity = features.reduce((sum, f, i) =>
            sum + f * this.weights.complexityWeights[i], 0);
        const technical = features.reduce((sum, f, i) =>
            sum + f * this.weights.technicalWeights[i], 0);

        this.lastPrediction = {
            complexity: Math.max(0, Math.min(1, complexity + 0.5)),
            technical: Math.max(0, Math.min(1, technical + 0.5))
        };

        return this.lastPrediction;
    }

    learn(question, answer) {
        if (!this.lastPrediction) {
            this.predict(question);
        }

        const features = this.extractFeatures(question);
        const words = answer.split(/\s+/);

        const actualComplexity = Math.min(1, words.length / 100);
        const actualTechnical = this.countPatterns(answer,
            ['algorithm', 'system', 'function', 'process', 'method', 'structure']) / Math.max(words.length, 1) * 10;

        const complexityError = Math.abs(actualComplexity - this.lastPrediction.complexity);
        const technicalError = Math.abs(actualTechnical - this.lastPrediction.technical);

        this.lastError = (complexityError + technicalError) / 2;

        for (let i = 0; i < features.length; i++) {
            this.weights.complexityWeights[i] += this.learningRate *
                (actualComplexity - this.lastPrediction.complexity) * features[i];
            this.weights.technicalWeights[i] += this.learningRate *
                (actualTechnical - this.lastPrediction.technical) * features[i];
        }

        return this.lastError;
    }

    getSurprise() {
        return this.lastError;
    }
}

// ============================================
// APPLICATION STATE
// ============================================

const state = {
    connected: false,
    currentDepth: 0,
    totalBranches: 0,
    heartedCount: 0,
    trajectory: [],
    currentNode: null,
    seedArchiveSource: [...SEED_ARCHIVE_FALLBACK],
    seedArchivePool: [],
    seedArchiveCursor: 0,

    curiosityNet: new CuriosityNet(),
    noveltyArchive: new NoveltyArchive(),
    predictionModel: new PredictionModel(),

    currentNovelty: 0,
    currentSurprise: 0,
    informationGain: 0,

    curiosityOutput: {
        depthDrive: 0.5,
        breadthDrive: 0.5,
        abstractDrive: 0.5,
        mechanismDrive: 0.5,
        explorationBonus: 0.5,
        questionLength: -0.3
    },

    promptEvolution: {
        generation: 0,
        activeGenome: null,
        prefetchedGenome: null,
        prefetchPromise: null,
        nextOperator: null,
        population: [],
        lineage: [],
        contextPool: [],
        mutationPromptPool: [...PROMPTBREEDER_MUTATION_PROMPTS]
    }
};

let seedIntroCleanupTimer = null;
let seedIntroRunId = 0;

function playContextQuestionIntro(questionText) {
    const questionEl = document.getElementById('current-question');
    if (!questionEl) return;

    questionEl.classList.remove('magic-intro');
    questionEl.textContent = questionText;
    void questionEl.offsetWidth;
    questionEl.classList.add('magic-intro');
}

function stopContextQuestionIntro() {
    const questionEl = document.getElementById('current-question');
    if (!questionEl) return;
    questionEl.classList.remove('magic-intro');
}

function prepareAnswerReveal() {
    const answerEl = document.getElementById('answer-text');
    const answerContainer = document.getElementById('current-answer');

    if (answerEl) {
        answerEl.classList.remove('answer-reveal');
        answerEl.textContent = '';
    }

    if (answerContainer) {
        answerContainer.classList.add('answer-hidden');
        answerContainer.classList.add('answer-loading');
    }
}

function revealAnswerText(answerText) {
    const answerEl = document.getElementById('answer-text');
    const answerContainer = document.getElementById('current-answer');
    if (!answerEl) return;

    answerEl.textContent = answerText;
    requestAnimationFrame(() => {
        stopContextQuestionIntro();
        if (answerContainer) {
            answerContainer.classList.remove('answer-hidden');
            answerContainer.classList.remove('answer-loading');
        }
        answerEl.classList.remove('answer-reveal');
        void answerEl.offsetWidth;
        answerEl.classList.add('answer-reveal');
    });
}

function setFollowUpAreaVisible(visible) {
    const chamber = document.getElementById('question-chamber');
    const custom = document.getElementById('explore-custom-question');

    if (chamber) {
        chamber.style.display = visible ? '' : 'none';
    }

    if (custom) {
        custom.style.display = visible ? 'flex' : 'none';
    }
}

function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function chooseRandom(items) {
    if (!Array.isArray(items) || items.length === 0) return null;
    return items[Math.floor(Math.random() * items.length)];
}

function clamp01(value) {
    return Math.max(0, Math.min(1, value));
}

function normalizeInstructionCandidate(text, fallback) {
    const normalized = parseMarkdown(text || '')
        .replace(/^[\-\*\d\.\)\s]+/, '')
        .replace(/^(instruction mutant|instruction|mutation prompt)\s*[:\-]\s*/i, '')
        .replace(/\s+/g, ' ')
        .trim()
        .replace(/^["'`]+|["'`]+$/g, '');

    if (!normalized) return fallback;
    return normalized.slice(0, 420);
}

function buildPromptGenome({
    taskPrompt,
    mutationPrompt,
    operator,
    parentGenome = null,
    contextExamples = null
}) {
    const safeTaskPrompt = normalizeInstructionCandidate(taskPrompt, chooseRandom(PROMPTBREEDER_SEED_TASK_PROMPTS));
    const safeMutationPrompt = normalizeInstructionCandidate(
        mutationPrompt,
        chooseRandom(PROMPTBREEDER_MUTATION_PROMPTS)
    );

    const lineage = parentGenome?.lineage
        ? [...parentGenome.lineage.slice(-5), safeTaskPrompt]
        : [safeTaskPrompt];

    return {
        id: `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
        taskPrompt: safeTaskPrompt,
        mutationPrompt: safeMutationPrompt,
        operator,
        createdAt: Date.now(),
        lineage,
        contextExamples: Array.isArray(contextExamples)
            ? contextExamples.slice(0, 3)
            : (parentGenome?.contextExamples || []).slice(0, 3),
        lastFitness: parentGenome?.lastFitness ?? 0.5
    };
}

function initializePromptEvolution() {
    if (state.promptEvolution.activeGenome) {
        return state.promptEvolution.activeGenome;
    }

    const seedGenome = buildPromptGenome({
        taskPrompt: chooseRandom(PROMPTBREEDER_SEED_TASK_PROMPTS),
        mutationPrompt: chooseRandom(PROMPTBREEDER_MUTATION_PROMPTS),
        operator: 'seed_init'
    });

    state.promptEvolution.activeGenome = seedGenome;
    state.promptEvolution.lineage = [...seedGenome.lineage];
    state.promptEvolution.population = [{
        taskPrompt: seedGenome.taskPrompt,
        mutationPrompt: seedGenome.mutationPrompt,
        fitness: 0.5,
        createdAt: seedGenome.createdAt
    }];

    return seedGenome;
}

function computePromptFitnessProxy(questions) {
    const novelty = clamp01(state.currentNovelty || 0);
    const surprise = clamp01(state.currentSurprise || 0);
    const exploration = clamp01((state.curiosityOutput?.explorationBonus || 0.5));
    const coverage = clamp01((questions?.length || 0) / CONFIG.numQuestions);

    const averageLength = (questions || []).reduce((sum, q) => sum + q.split(/\s+/).length, 0) / Math.max(questions?.length || 1, 1);
    const lengthScore = clamp01(1 - Math.abs(averageLength - 8) / 12);

    return clamp01(
        0.18 +
        0.30 * novelty +
        0.22 * surprise +
        0.18 * exploration +
        0.08 * coverage +
        0.04 * lengthScore
    );
}

function recordPromptOutcome(genome, fitness, currentAnswer) {
    if (!genome) return;

    genome.lastFitness = fitness;
    state.promptEvolution.activeGenome = genome;

    state.promptEvolution.population.push({
        taskPrompt: genome.taskPrompt,
        mutationPrompt: genome.mutationPrompt,
        fitness,
        createdAt: Date.now()
    });

    state.promptEvolution.population.sort((a, b) => b.fitness - a.fitness);
    if (state.promptEvolution.population.length > PROMPTBREEDER_HISTORY_LIMIT) {
        state.promptEvolution.population = state.promptEvolution.population.slice(0, PROMPTBREEDER_HISTORY_LIMIT);
    }

    state.promptEvolution.lineage = [...(genome.lineage || [genome.taskPrompt])].slice(-8);

    if (genome.mutationPrompt && !state.promptEvolution.mutationPromptPool.includes(genome.mutationPrompt)) {
        state.promptEvolution.mutationPromptPool.push(genome.mutationPrompt);
        if (state.promptEvolution.mutationPromptPool.length > 40) {
            state.promptEvolution.mutationPromptPool = state.promptEvolution.mutationPromptPool.slice(-40);
        }
    }

    if (typeof currentAnswer === 'string' && currentAnswer.trim().length > 0) {
        state.promptEvolution.contextPool.push(currentAnswer.trim());
        if (state.promptEvolution.contextPool.length > 14) {
            state.promptEvolution.contextPool = state.promptEvolution.contextPool.slice(-14);
        }
    }
}

function weightedPickPopulationMember(population, excludeTaskPrompt = null) {
    const filtered = population.filter(p => p.taskPrompt !== excludeTaskPrompt);
    if (filtered.length === 0) return null;

    const totalWeight = filtered.reduce((sum, item) => sum + Math.max(0.01, item.fitness || 0), 0);
    let threshold = Math.random() * totalWeight;

    for (const item of filtered) {
        threshold -= Math.max(0.01, item.fitness || 0);
        if (threshold <= 0) {
            return item;
        }
    }

    return filtered[filtered.length - 1];
}

async function callPromptMutationModel(userPrompt, fallback = '') {
    if (!state.connected) {
        return fallback;
    }

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: CONFIG.model,
                max_tokens: 320,
                messages: [{ role: 'user', content: userPrompt }]
            })
        });

        if (!response.ok) {
            return fallback;
        }

        const data = await response.json();
        const raw = data.content?.[0]?.text?.trim() || fallback;
        return normalizeInstructionCandidate(raw, fallback);
    } catch (error) {
        console.warn('Prompt mutation call failed:', error);
        return fallback;
    }
}

function fallbackGenomeMutation(parentGenome, operator) {
    const fallbackTaskPrompt = normalizeInstructionCandidate(
        `${parentGenome.taskPrompt} Focus on novelty and clear wording.`,
        parentGenome.taskPrompt
    );

    return buildPromptGenome({
        taskPrompt: fallbackTaskPrompt,
        mutationPrompt: parentGenome.mutationPrompt,
        operator,
        parentGenome
    });
}

async function mutateDirectFirstOrder(parentGenome) {
    const userPrompt = `${parentGenome.mutationPrompt}

INSTRUCTION: ${parentGenome.taskPrompt}
INSTRUCTION MUTANT:`;

    const taskPrompt = await callPromptMutationModel(userPrompt, parentGenome.taskPrompt);
    return buildPromptGenome({
        taskPrompt,
        mutationPrompt: parentGenome.mutationPrompt,
        operator: 'direct_first_order',
        parentGenome
    });
}

async function mutateDirectZeroOrder(parentGenome) {
    const thinkingStyle = chooseRandom(PROMPTBREEDER_THINKING_STYLES) || '';
    const userPrompt = `${PROMPTBREEDER_TASK_DESCRIPTION}
Thinking style: ${thinkingStyle}
A list of 100 hints:
1.`;

    const taskPrompt = await callPromptMutationModel(userPrompt, parentGenome.taskPrompt);
    return buildPromptGenome({
        taskPrompt,
        mutationPrompt: chooseRandom(state.promptEvolution.mutationPromptPool) || parentGenome.mutationPrompt,
        operator: 'direct_zero_order',
        parentGenome
    });
}

async function mutateEDAPopulation(parentGenome) {
    const population = shuffleArray(state.promptEvolution.population.map(p => p.taskPrompt)).slice(0, 8);
    const promptList = population.map((p, i) => `${i + 1}. ${p}`).join('\n');

    const userPrompt = `You are evolving instruction prompts for this task:
${PROMPTBREEDER_TASK_DESCRIPTION}

Here is a diverse list of current instructions:
${promptList}

Continue the list with one NEW instruction that is not a copy:
${population.length + 1}.`;

    const taskPrompt = await callPromptMutationModel(userPrompt, parentGenome.taskPrompt);
    return buildPromptGenome({
        taskPrompt,
        mutationPrompt: parentGenome.mutationPrompt,
        operator: 'eda_population',
        parentGenome
    });
}

async function mutateEDARanked(parentGenome) {
    const ranked = [...state.promptEvolution.population]
        .sort((a, b) => a.fitness - b.fitness)
        .slice(0, 8);
    const rankedList = ranked.map((p, i) => `${i + 1}. ${p.taskPrompt}`).join('\n');

    const userPrompt = `INSTRUCTION: ${parentGenome.mutationPrompt}
A List of Responses in descending order of score. ${ranked.length + 1} is the best response. It resembles ${ranked.length} more than it does (1).
${rankedList}

${ranked.length + 1}.`;

    const taskPrompt = await callPromptMutationModel(userPrompt, parentGenome.taskPrompt);
    return buildPromptGenome({
        taskPrompt,
        mutationPrompt: parentGenome.mutationPrompt,
        operator: 'eda_ranked',
        parentGenome
    });
}

async function mutateLineage(parentGenome) {
    const lineage = (state.promptEvolution.lineage.length > 0 ? state.promptEvolution.lineage : parentGenome.lineage)
        .slice(-8);
    const lineageList = lineage.map((p, i) => `${i + 1}. ${p}`).join('\n');

    const userPrompt = `GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY
${lineageList}

Generate the next improved instruction:`;

    const taskPrompt = await callPromptMutationModel(userPrompt, parentGenome.taskPrompt);
    return buildPromptGenome({
        taskPrompt,
        mutationPrompt: parentGenome.mutationPrompt,
        operator: 'eda_lineage',
        parentGenome
    });
}

async function mutateHyperZeroOrder(parentGenome) {
    const thinkingStyle = chooseRandom(PROMPTBREEDER_THINKING_STYLES) || '';
    const newMutationPrompt = await callPromptMutationModel(
        `${PROMPTBREEDER_TASK_DESCRIPTION}
Thinking style: ${thinkingStyle}
Generate one mutation instruction that rewrites another instruction to produce better follow-up questions.`,
        parentGenome.mutationPrompt
    );

    const taskPrompt = await callPromptMutationModel(
        `${newMutationPrompt}

INSTRUCTION: ${parentGenome.taskPrompt}
INSTRUCTION MUTANT:`,
        parentGenome.taskPrompt
    );

    return buildPromptGenome({
        taskPrompt,
        mutationPrompt: newMutationPrompt,
        operator: 'hyper_zero_order',
        parentGenome
    });
}

async function mutateHyperFirstOrder(parentGenome) {
    const improvedMutationPrompt = await callPromptMutationModel(
        `${PROMPTBREEDER_HYPER_MUTATION_PROMPT}
${parentGenome.mutationPrompt}
Improved instruction:`,
        parentGenome.mutationPrompt
    );

    const taskPrompt = await callPromptMutationModel(
        `${improvedMutationPrompt}

INSTRUCTION: ${parentGenome.taskPrompt}
INSTRUCTION MUTANT:`,
        parentGenome.taskPrompt
    );

    return buildPromptGenome({
        taskPrompt,
        mutationPrompt: improvedMutationPrompt,
        operator: 'hyper_first_order',
        parentGenome
    });
}

async function mutateLamarckian(parentGenome, mutationContext) {
    const workingOut = mutationContext.currentAnswer || chooseRandom(state.promptEvolution.contextPool) || '';
    if (!workingOut) {
        return mutateDirectFirstOrder(parentGenome);
    }

    const userPrompt = `I gave a friend an instruction and some advice. Here are the correct examples of his workings out:
${workingOut}
The instruction was:`;

    const taskPrompt = await callPromptMutationModel(userPrompt, parentGenome.taskPrompt);
    return buildPromptGenome({
        taskPrompt,
        mutationPrompt: parentGenome.mutationPrompt,
        operator: 'lamarckian',
        parentGenome
    });
}

async function mutateCrossoverContextShuffle(parentGenome) {
    const donor = weightedPickPopulationMember(state.promptEvolution.population, parentGenome.taskPrompt);
    const donorTaskPrompt = donor?.taskPrompt || chooseRandom(PROMPTBREEDER_SEED_TASK_PROMPTS) || parentGenome.taskPrompt;

    const blendedTaskPrompt = await callPromptMutationModel(
        `Combine the strengths of these two instruction prompts into one concise improved instruction.

Parent instruction:
${parentGenome.taskPrompt}

Donor instruction:
${donorTaskPrompt}

Improved instruction:`,
        parentGenome.taskPrompt
    );

    const shuffledContexts = shuffleArray(state.promptEvolution.contextPool).slice(0, 3);
    return buildPromptGenome({
        taskPrompt: blendedTaskPrompt,
        mutationPrompt: parentGenome.mutationPrompt,
        operator: 'crossover_context_shuffle',
        parentGenome,
        contextExamples: shuffledContexts
    });
}

async function applyPromptMutationOperator(operator, parentGenome, mutationContext) {
    try {
        switch (operator) {
            case 'direct_first_order':
                return await mutateDirectFirstOrder(parentGenome);
            case 'direct_zero_order':
                return await mutateDirectZeroOrder(parentGenome);
            case 'eda_population':
                return await mutateEDAPopulation(parentGenome);
            case 'eda_ranked':
                return await mutateEDARanked(parentGenome);
            case 'eda_lineage':
                return await mutateLineage(parentGenome);
            case 'hyper_zero_order':
                return await mutateHyperZeroOrder(parentGenome);
            case 'hyper_first_order':
                return await mutateHyperFirstOrder(parentGenome);
            case 'lamarckian':
                return await mutateLamarckian(parentGenome, mutationContext);
            case 'crossover_context_shuffle':
                return await mutateCrossoverContextShuffle(parentGenome);
            default:
                return await mutateDirectFirstOrder(parentGenome);
        }
    } catch (error) {
        console.warn(`Mutation operator "${operator}" failed:`, error);
        return fallbackGenomeMutation(parentGenome, operator);
    }
}

function schedulePromptMutationPrefetch(mutationContext, parentGenomeOverride = null) {
    initializePromptEvolution();

    if (state.promptEvolution.prefetchPromise) {
        return state.promptEvolution.prefetchPromise;
    }

    const parentGenome = parentGenomeOverride || state.promptEvolution.activeGenome;
    if (!parentGenome) return null;

    const selectedOperator = chooseRandom(PROMPTBREEDER_OPERATORS) || 'direct_first_order';
    state.promptEvolution.nextOperator = selectedOperator;

    state.promptEvolution.prefetchPromise = (async () => {
        const mutatedGenome = await applyPromptMutationOperator(selectedOperator, parentGenome, mutationContext);
        state.promptEvolution.prefetchedGenome = mutatedGenome;
        state.promptEvolution.prefetchPromise = null;
        return mutatedGenome;
    })().catch((error) => {
        console.warn('Prefetch mutation failed:', error);
        state.promptEvolution.prefetchedGenome = fallbackGenomeMutation(parentGenome, selectedOperator);
        state.promptEvolution.prefetchPromise = null;
        return state.promptEvolution.prefetchedGenome;
    });

    return state.promptEvolution.prefetchPromise;
}

function consumePromptGenomeForGeneration(mutationContext) {
    initializePromptEvolution();

    const readyGenome = state.promptEvolution.prefetchedGenome || state.promptEvolution.activeGenome;
    state.promptEvolution.prefetchedGenome = null;
    state.promptEvolution.activeGenome = readyGenome;
    state.promptEvolution.generation += 1;

    // Precompute the next generation's mutation in the background.
    schedulePromptMutationPrefetch(mutationContext, readyGenome);

    return readyGenome;
}

function ensurePromptMutationPipeline(mutationContext = { currentQuestion: '', currentAnswer: '', styleGuidance: '' }) {
    initializePromptEvolution();
    if (!state.connected) return;
    if (state.promptEvolution.prefetchedGenome || state.promptEvolution.prefetchPromise) return;
    schedulePromptMutationPrefetch(mutationContext, state.promptEvolution.activeGenome);
}

function composeFollowUpGenerationPrompt({
    genome,
    currentQuestion,
    currentAnswer,
    styleGuidance,
    targetWords
}) {
    const contextExamples = (genome.contextExamples || [])
        .slice(0, 2)
        .map((example, idx) => `Example ${idx + 1} successful working style:\n${example}`)
        .join('\n\n');

    const optionalExamples = contextExamples
        ? `\n\nReference examples from successful prior outputs:\n${contextExamples}\n`
        : '';

    return `${genome.taskPrompt}

Current Q&A:
Q: "${currentQuestion}"
A: "${currentAnswer}"
${optionalExamples}
Generate ${CONFIG.numQuestions} follow-up questions.
Each question must stay around ${targetWords} words and include a question mark.
Keep questions diverse and avoid repeating wording from each other.

Style guidance:
${styleGuidance}

Format: return exactly ${CONFIG.numQuestions} numbered lines (1., 2., ...).`;
}

async function readApiErrorMessage(response, fallback, maxLength = 160) {
    let raw = '';
    try {
        raw = await response.text();
    } catch (error) {
        return fallback;
    }

    if (!raw) {
        return fallback;
    }

    try {
        const parsed = JSON.parse(raw);
        const msg = parsed?.error?.message || parsed?.message;
        if (msg && typeof msg === 'string') {
            return msg.slice(0, maxLength);
        }
    } catch (error) {
        // Non-JSON body; fall back to raw snippet.
    }

    return raw.slice(0, maxLength);
}

// ============================================
// API CALLS (via Vercel serverless functions)
// ============================================

async function checkConnection() {
    initializePromptEvolution();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: CONFIG.model,
                max_tokens: 10,
                messages: [{ role: 'user', content: 'Hi' }]
            })
        });

        if (response.ok) {
            state.connected = true;
            updateConnectionStatus(true);
            ensurePromptMutationPipeline();
            return true;
        } else if (response.status === 429) {
            state.connected = true;
            updateConnectionStatus(true, 'Rate limited - wait a moment');
            ensurePromptMutationPipeline();
            return true;
        } else {
            const message = await readApiErrorMessage(response, `HTTP ${response.status}`, 140);
            console.error('API error:', message);
            state.connected = false;
            updateConnectionStatus(false, message || 'API Error');
            return false;
        }
    } catch (e) {
        console.error('Connection failed:', e);
        state.connected = false;
        updateConnectionStatus(false, 'Connection Failed');
        return false;
    }
}

function updateConnectionStatus(connected, statusMsg = '') {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');

    if (connected) {
        dot.classList.add('connected');
        text.textContent = statusMsg || 'Connected';
    } else {
        dot.classList.remove('connected');
        text.textContent = statusMsg || 'Disconnected';
    }
}

function shuffleArray(items) {
    const arr = [...items];
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
}

async function loadSeedArchive() {
    try {
        const response = await fetch('/seed-archive.json', { cache: 'no-store' });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const payload = await response.json();
        if (!Array.isArray(payload)) {
            throw new Error('Archive payload was not an array');
        }

        const normalized = payload
            .map(item => typeof item === 'string' ? item.trim() : '')
            .filter(item => item.length > 0);

        if (normalized.length > 0) {
            state.seedArchiveSource = normalized;
        }
    } catch (error) {
        console.warn('Using fallback archive list:', error);
        state.seedArchiveSource = [...SEED_ARCHIVE_FALLBACK];
    }

    state.seedArchivePool = shuffleArray(state.seedArchiveSource);
    state.seedArchiveCursor = 0;
}

function takeSeedArchiveBatch() {
    const source = state.seedArchiveSource.length > 0
        ? state.seedArchiveSource
        : SEED_ARCHIVE_FALLBACK;

    if (source.length <= SEED_ARCHIVE_PAGE_SIZE) {
        return [...source];
    }

    if (state.seedArchivePool.length === 0 || state.seedArchiveCursor >= state.seedArchivePool.length) {
        state.seedArchivePool = shuffleArray(source);
        state.seedArchiveCursor = 0;
    }

    const batch = [];
    while (batch.length < SEED_ARCHIVE_PAGE_SIZE) {
        if (state.seedArchiveCursor >= state.seedArchivePool.length) {
            state.seedArchivePool = shuffleArray(source);
            state.seedArchiveCursor = 0;
        }
        batch.push(state.seedArchivePool[state.seedArchiveCursor]);
        state.seedArchiveCursor += 1;
    }

    return batch;
}

function renderSeedArchiveBatch() {
    const container = document.getElementById('seed-options');
    if (!container) return;

    const batch = takeSeedArchiveBatch();
    container.innerHTML = '';

    batch.forEach((seed, index) => {
        const option = document.createElement('button');
        option.type = 'button';
        option.className = 'seed-option';
        option.dataset.seed = seed;
        option.textContent = seed;
        option.classList.add('reveal-card');
        option.style.setProperty('--reveal-delay', `${index * 70}ms`);
        container.appendChild(option);
    });
}

async function presentSeedArchiveWithIntro(archiveLoadPromise = null) {
    const runId = ++seedIntroRunId;
    const title = document.getElementById('start-title');
    const container = document.getElementById('seed-options');

    if (!container) {
        renderSeedArchiveBatch();
        return;
    }

    if (seedIntroCleanupTimer) {
        clearTimeout(seedIntroCleanupTimer);
    }

    container.classList.remove('is-visible');
    container.classList.add('is-staged');
    container.innerHTML = '';

    if (title) {
        title.classList.remove('magic-intro');
        void title.offsetWidth;
        title.classList.add('magic-intro');
    }

    const revealDelayPromise = wait(60);
    const dataReadyPromise = archiveLoadPromise || Promise.resolve();
    await Promise.all([revealDelayPromise, dataReadyPromise]);

    if (runId !== seedIntroRunId) return;

    renderSeedArchiveBatch();
    container.classList.remove('is-staged');
    container.classList.add('is-visible');

    seedIntroCleanupTimer = setTimeout(() => {
        if (title && runId === seedIntroRunId) {
            title.classList.remove('magic-intro');
        }
    }, 900);
}

function hideAnswerAndFollowUps() {
    prepareAnswerReveal();
    setFollowUpAreaVisible(false);
}

async function generateResponseStreaming(prompt, context = '', onChunk) {
    if (!state.connected) {
        throw new Error('Not connected to API');
    }

    const systemPrompt = `You are a warm, knowledgeable guide helping someone explore ideas through curiosity-driven learning. Imagine you are a wise and friendly mentor in an old library.
Your responses should be:
- Concise (${CONFIG.responseSentences} sentences max)
- Insightful and thought-provoking
- Accurate but accessible and warm in tone
- End with an implicit hook that invites further questions
- NEVER use em-dashes

Context of the exploration so far:
${context}`;

    const response = await fetch('/api/chat-stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model: CONFIG.model,
            max_tokens: 500,
            system: systemPrompt,
            messages: [{ role: 'user', content: prompt }]
        })
    });

    if (!response.ok) {
        const errorText = await response.text();
        console.error('Streaming error:', errorText);
        throw new Error(`Generation failed: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const jsonStr = line.slice(6).trim();
                if (jsonStr && jsonStr !== '[DONE]') {
                    try {
                        const json = JSON.parse(jsonStr);
                        if (json.type === 'content_block_delta' && json.delta?.text) {
                            fullText += json.delta.text;
                            if (onChunk) onChunk(parseMarkdown(fullText));
                        }
                    } catch (e) {
                        // Ignore parse errors
                    }
                }
            }
        }
    }

    return parseMarkdown(fullText.trim()) || 'No response generated';
}

async function generateResponse(prompt, context = '') {
    if (!state.connected) {
        throw new Error('Not connected to API');
    }

    const systemPrompt = `You are a warm, knowledgeable guide helping someone explore ideas through curiosity-driven learning. Imagine you are a wise and friendly mentor in an old library.
Your responses should be:
- Concise (${CONFIG.responseSentences} sentences max)
- Insightful and thought-provoking
- Accurate but accessible and warm in tone
- End with an implicit hook that invites further questions
- NEVER use em-dashes

Context of the exploration so far:
${context}`;

    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model: CONFIG.model,
            max_tokens: 500,
            system: systemPrompt,
            messages: [{ role: 'user', content: prompt }]
        })
    });

    if (!response.ok) {
        const errorMessage = await readApiErrorMessage(response, `HTTP ${response.status}`, 180);
        throw new Error(`Generation failed: ${errorMessage}`);
    }

    const data = await response.json();
    const rawText = data.content?.[0]?.text?.trim() || 'No response generated';
    return parseMarkdown(rawText);
}

// ============================================
// NEURAL QUESTION GENERATION
// ============================================

async function generateQuestionsWithCuriosity(currentQuestion, currentAnswer, trajectory) {
    if (!state.connected) {
        throw new Error('Not connected to API');
    }

    const qWords = currentQuestion.toLowerCase().split(/\s+/);
    const networkInputs = {
        wordFreq: Math.min(1, qWords.length / 15),
        qType: calculateQuestionType(currentQuestion),
        abstract: calculateAbstractness(currentQuestion),
        novelty: state.currentNovelty,
        depth: Math.min(1, state.currentDepth / 10),
        predErr: state.currentSurprise
    };

    state.curiosityOutput = state.curiosityNet.forward(networkInputs);

    const { depthDrive, breadthDrive, abstractDrive, mechanismDrive, explorationBonus, questionLength } = state.curiosityOutput;

    // Convert questionLength (-1 to 1) to word target (4-12 words)
    const lengthScore = (questionLength + 1) / 2; // 0 to 1
    const targetWords = Math.round(4 + lengthScore * 8); // 4-12 words

    let styleGuidance = '';

    if (questionLength < -0.3) {
        styleGuidance += `- Keep questions SHORT and punchy (around ${targetWords} words max). Be direct.\n`;
    } else if (questionLength < 0.3) {
        styleGuidance += `- Keep questions concise (around ${targetWords} words). Clear and readable.\n`;
    } else {
        styleGuidance += `- Questions can be more detailed (up to ${targetWords} words) if needed for nuance.\n`;
    }

    if (depthDrive > breadthDrive) {
        styleGuidance += '- Go DEEPER into the current topic\n';
    } else {
        styleGuidance += '- BRANCH OUT to related but different topics\n';
    }

    if (abstractDrive > 0.5) {
        styleGuidance += '- Ask about principles and concepts\n';
    } else {
        styleGuidance += '- Ask about examples and practical cases\n';
    }

    if (mechanismDrive > 0.5) {
        styleGuidance += '- Focus on "how" questions\n';
    } else {
        styleGuidance += '- Focus on "why" questions\n';
    }

    if (explorationBonus > 0.6) {
        styleGuidance += '- Include a surprising angle\n';
    }

    if (state.noveltyArchive.archive.length > 5) {
        const recentTopics = state.noveltyArchive.archive.slice(-5).map(a => a.question.substring(0, 30)).join(', ');
        styleGuidance += `- Avoid: ${recentTopics}\n`;
    }

    const mutationContext = {
        currentQuestion,
        currentAnswer,
        styleGuidance,
        targetWords
    };

    const genome = consumePromptGenomeForGeneration(mutationContext);
    const prompt = composeFollowUpGenerationPrompt({
        genome,
        currentQuestion,
        currentAnswer,
        styleGuidance,
        targetWords
    });

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: CONFIG.model,
                max_tokens: 600,
                messages: [{ role: 'user', content: prompt }]
            })
        });

        if (!response.ok) {
            throw new Error(`Question generation failed: ${response.status}`);
        }

        const data = await response.json();
        const responseText = parseMarkdown(data.content?.[0]?.text || '');
        const lines = responseText.trim().split('\n');
        const questions = [];

        for (const line of lines) {
            const cleaned = line
                .replace(/^\s*[\-\*\u2022]?\s*\d*[\.\)]?\s*/, '')
                .replace(/\*\*/g, '')
                .trim();
            if (cleaned.length > 8 && cleaned.includes('?')) {
                questions.push(cleaned);
            }
        }

        if (questions.length < CONFIG.numQuestions) {
            const sentenceSplits = responseText
                .split(/\?(?:\s+|$)/)
                .map(fragment => fragment.trim())
                .filter(Boolean);
            for (const fragment of sentenceSplits) {
                if (questions.length >= CONFIG.numQuestions) break;
                const candidate = `${fragment.replace(/^[\-\*\d\.\)\s]+/, '').trim()}?`;
                if (candidate.length > 8 && !questions.includes(candidate)) {
                    questions.push(candidate);
                }
            }
        }

        while (questions.length < CONFIG.numQuestions) {
            questions.push(`What else would you like to know about ${currentQuestion.split(' ').slice(0, 4).join(' ')}...?`);
        }

        const finalQuestions = questions.slice(0, CONFIG.numQuestions);
        const fitness = computePromptFitnessProxy(finalQuestions);
        recordPromptOutcome(genome, fitness, currentAnswer);
        return finalQuestions;

    } catch (error) {
        console.error('Question generation error:', error);
        const fallbackQuestions = Array(CONFIG.numQuestions).fill(
            `What else would you like to explore about ${currentQuestion.split(' ').slice(0, 4).join(' ')}...?`
        );
        recordPromptOutcome(genome, 0.25, currentAnswer);
        return fallbackQuestions;
    }
}

function calculateQuestionType(question) {
    const q = question.toLowerCase();
    if (q.includes('why') || q.includes('reason')) return 0.2;
    if (q.includes('what') || q.includes('define')) return 0.4;
    if (q.includes('how') || q.includes('process')) return 0.8;
    if (q.includes('when') || q.includes('where')) return 0.6;
    return 0.5;
}

function calculateAbstractness(question) {
    const abstractWords = ['concept', 'theory', 'principle', 'nature', 'meaning', 'essence', 'fundamental'];
    const concreteWords = ['example', 'specific', 'case', 'instance', 'practical', 'real', 'actual'];

    const q = question.toLowerCase();
    const abstractCount = abstractWords.filter(w => q.includes(w)).length;
    const concreteCount = concreteWords.filter(w => q.includes(w)).length;

    if (abstractCount + concreteCount === 0) return 0.5;
    return abstractCount / (abstractCount + concreteCount);
}

// ============================================
// VISUALIZATION (warm parchment colors)
// ============================================

function drawNetworkVisualization() {
    const canvas = document.getElementById('network-canvas');
    const ctx = canvas.getContext('2d');

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = '#e8dece';
    ctx.fillRect(0, 0, width, height);

    const net = state.curiosityNet;
    const nodes = Array.from(net.nodes.values());
    const connections = net.connections;

    const inputNodes = nodes.filter(n => n.type === 'input');
    const hiddenNodes = nodes.filter(n => n.type === 'hidden');
    const outputNodes = nodes.filter(n => n.type === 'output');

    const positions = new Map();

    inputNodes.forEach((n, i) => {
        positions.set(n.id, {
            x: 40,
            y: 30 + i * (height - 60) / Math.max(inputNodes.length - 1, 1)
        });
    });

    hiddenNodes.forEach((n, i) => {
        positions.set(n.id, {
            x: width / 2,
            y: 30 + i * (height - 60) / Math.max(hiddenNodes.length, 1)
        });
    });

    outputNodes.forEach((n, i) => {
        positions.set(n.id, {
            x: width - 40,
            y: 30 + i * (height - 60) / Math.max(outputNodes.length - 1, 1)
        });
    });

    // Draw connections with warm colors
    for (const conn of connections) {
        if (!conn.enabled) continue;

        const from = positions.get(conn.from);
        const to = positions.get(conn.to);

        if (!from || !to) continue;

        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);

        const intensity = Math.abs(conn.weight);
        if (conn.weight > 0) {
            ctx.strokeStyle = `rgba(90, 110, 58, ${0.3 + intensity * 0.5})`; // olive green
        } else {
            ctx.strokeStyle = `rgba(181, 68, 62, ${0.3 + intensity * 0.4})`; // warm red
        }
        ctx.lineWidth = 1 + intensity;
        ctx.stroke();
    }

    // Draw nodes with warm colors
    for (const node of nodes) {
        const pos = positions.get(node.id);
        if (!pos) continue;

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, node.type === 'hidden' ? 6 : 8, 0, Math.PI * 2);

        if (node.type === 'input') {
            ctx.fillStyle = '#7a4a1e'; // warm brown
        } else if (node.type === 'output') {
            ctx.fillStyle = '#a06830'; // amber
        } else {
            ctx.fillStyle = '#5a6e3a'; // olive
        }
        ctx.fill();

        // Add subtle border
        ctx.strokeStyle = 'rgba(59, 46, 26, 0.3)';
        ctx.lineWidth = 1;
        ctx.stroke();

        if (node.type !== 'hidden') {
            ctx.fillStyle = '#8a7a5a';
            ctx.font = '8px Georgia, serif';
            ctx.textAlign = node.type === 'input' ? 'right' : 'left';
            const labelX = node.type === 'input' ? pos.x - 12 : pos.x + 12;
            ctx.fillText(node.label.substring(0, 6), labelX, pos.y + 3);
        }
    }

    document.getElementById('node-count').textContent = nodes.length;
    document.getElementById('connection-count').textContent = connections.filter(c => c.enabled).length;
    document.getElementById('mutation-count').textContent = net.mutationCount;
}

function drawArchiveVisualization() {
    const canvas = document.getElementById('archive-canvas');
    const ctx = canvas.getContext('2d');

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = '#e8dece';
    ctx.fillRect(0, 0, width, height);

    const data = state.noveltyArchive.getVisualizationData();

    if (data.length === 0) {
        ctx.fillStyle = '#8a7a5a';
        ctx.font = '12px Georgia, serif';
        ctx.textAlign = 'center';
        ctx.fillText('Your journey awaits...', width / 2, height / 2);
        return;
    }

    // Axes in warm muted color
    ctx.strokeStyle = '#c4b08a';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(width / 2, 10);
    ctx.lineTo(width / 2, height - 10);
    ctx.moveTo(10, height / 2);
    ctx.lineTo(width - 10, height / 2);
    ctx.stroke();

    const padding = 20;
    for (const point of data) {
        const x = padding + (point.x + 1) / 2 * (width - 2 * padding);
        const y = padding + (1 - (point.y + 1) / 2) * (height - 2 * padding);

        ctx.beginPath();
        ctx.arc(x, y, 3 + point.novelty * 4, 0, Math.PI * 2);

        // Warm amber tones based on novelty
        const r = Math.round(122 + point.novelty * 80);
        const g = Math.round(74 + point.novelty * 30);
        const b = Math.round(30);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.7)`;
        ctx.fill();

        // Subtle border
        ctx.strokeStyle = 'rgba(59, 46, 26, 0.2)';
        ctx.lineWidth = 0.5;
        ctx.stroke();
    }

    ctx.fillStyle = '#8a7a5a';
    ctx.font = '9px Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('Abstract', width / 2, 12);
    ctx.fillText('Concrete', width / 2, height - 4);
    ctx.textAlign = 'left';
    ctx.fillText('Why', 4, height / 2 - 4);
    ctx.textAlign = 'right';
    ctx.fillText('How', width - 4, height / 2 - 4);

    document.getElementById('archive-size').textContent = state.noveltyArchive.archive.length;
    document.getElementById('frontiers-found').textContent = state.noveltyArchive.frontiersFound;
}

function updateCuriositySignals() {
    // Backend signals still computed but not displayed in UI
    // (novelty score, prediction error, info gain bars removed from HTML)

    // Update curiosity dimension bars in sidebar
    const out = state.curiosityOutput;
    document.getElementById('depth-breadth-bar').style.width = `${out.depthDrive * 50 + 25}%`;
    document.getElementById('depth-breadth-val').textContent = out.depthDrive.toFixed(2);

    document.getElementById('abstract-concrete-bar').style.width = `${out.abstractDrive * 50 + 25}%`;
    document.getElementById('abstract-concrete-val').textContent = out.abstractDrive.toFixed(2);

    document.getElementById('why-how-bar').style.width = `${out.mechanismDrive * 50 + 25}%`;
    document.getElementById('why-how-val').textContent = out.mechanismDrive.toFixed(2);

    document.getElementById('exploration-bar').style.width = `${out.explorationBonus * 50 + 25}%`;
    document.getElementById('exploration-val').textContent = out.explorationBonus.toFixed(2);
}

function updateTrajectoryList() {
    const list = document.getElementById('trajectory-list');
    list.innerHTML = '';

    state.trajectory.forEach((node, index) => {
        const item = document.createElement('div');
        item.className = 'trajectory-item';
        if (index === state.trajectory.length - 1) {
            item.classList.add('active');
        }
        if (node.hearted) {
            item.classList.add('hearted');
        }

        const noveltyClass = node.novelty > 0.6 ? 'high-novelty' : node.novelty < 0.3 ? 'low-novelty' : '';

        item.innerHTML = `
            <div class="trajectory-depth">Step ${node.depth + 1} <span class="novelty-badge ${noveltyClass}">${node.novelty.toFixed(2)}</span></div>
            <div>${truncate(node.question, 35)}</div>
        `;

        item.onclick = () => jumpToNode(index);
        list.appendChild(item);
    });

    list.scrollTop = list.scrollHeight;
}

function updateStats() {
    document.getElementById('depth-counter').textContent = state.currentDepth;
    document.getElementById('heart-counter').textContent = state.heartedCount;
}

function truncate(str, len) {
    if (str.length <= len) return str;
    return str.substring(0, len - 3) + '...';
}

function jumpToNode(index) {
    console.log('Jump to node:', index);
}

// ============================================
// MAIN EXPLORATION FLOW
// ============================================

async function startExploration(seedQuestion) {
    document.getElementById('start-screen').style.display = 'none';
    document.getElementById('exploration-view').style.display = 'block';
    document.getElementById('loading').style.display = 'none';

    playContextQuestionIntro(seedQuestion);
    hideAnswerAndFollowUps();
    document.getElementById('question-options').innerHTML = '';

    try {
        state.predictionModel.predict(seedQuestion);
        ensurePromptMutationPipeline({ currentQuestion: seedQuestion, currentAnswer: '', styleGuidance: '' });

        const answer = await generateResponse(
            `Question: ${seedQuestion}\n\nProvide a concise, insightful answer:`,
            ''
        );
        revealAnswerText(answer);
        const answerRevealTimestamp = Date.now();

        state.currentSurprise = state.predictionModel.learn(seedQuestion, answer);

        const context = '';
        const archiveResult = state.noveltyArchive.maybeAdd(seedQuestion, answer, context, CONFIG.noveltyThreshold);
        state.currentNovelty = archiveResult.novelty;

        state.informationGain = Math.min(1, answer.split(/\s+/).length / 50);

        const node = {
            question: seedQuestion,
            answer: answer,
            hearted: false,
            timestamp: Date.now(),
            depth: 0,
            novelty: state.currentNovelty,
            surprise: state.currentSurprise
        };

        state.trajectory.push(node);
        state.currentNode = node;
        state.currentDepth = 0;

        const heartBtn = document.getElementById('heart-btn');
        heartBtn.querySelector('.heart-icon').innerHTML = '&#10084;';
        heartBtn.classList.remove('active');

        updateTrajectoryList();
        updateStats();
        updateCuriositySignals();
        drawNetworkVisualization();
        drawArchiveVisualization();

        const questions = await generateQuestionsWithCuriosity(seedQuestion, answer, state.trajectory);
        const readDelay = Math.max(0, 5000 - (Date.now() - answerRevealTimestamp));
        if (readDelay > 0) {
            await wait(readDelay);
        }

        displayQuestionOptions(questions);
        setFollowUpAreaVisible(true);

        if (state.curiosityOutput.explorationBonus > 0.5 && Math.random() < CONFIG.mutationRate) {
            state.curiosityNet.mutate();
            drawNetworkVisualization();
        }

    } catch (error) {
        console.error('Exploration error:', error);
        stopContextQuestionIntro();
        alert('Error connecting to API. Please check configuration.\n\nError: ' + error.message);
        document.getElementById('start-screen').style.display = 'block';
        document.getElementById('exploration-view').style.display = 'none';
    }
}

async function selectQuestion(question) {
    playContextQuestionIntro(question);
    hideAnswerAndFollowUps();
    document.getElementById('question-options').innerHTML = '';

    try {
        const context = state.trajectory.slice(-3).map(t =>
            `Q: ${t.question}\nA: ${t.answer}`
        ).join('\n\n');

        state.predictionModel.predict(question);
        ensurePromptMutationPipeline({ currentQuestion: question, currentAnswer: '', styleGuidance: '' });

        const answer = await generateResponse(
            `Question: ${question}\n\nProvide a concise, insightful answer:`,
            context
        );
        revealAnswerText(answer);
        const answerRevealTimestamp = Date.now();

        state.currentSurprise = state.predictionModel.learn(question, answer);

        const archiveResult = state.noveltyArchive.maybeAdd(question, answer, context, CONFIG.noveltyThreshold);
        state.currentNovelty = archiveResult.novelty;

        state.informationGain = Math.min(1, answer.split(/\s+/).length / 50);

        const node = {
            question: question,
            answer: answer,
            hearted: false,
            timestamp: Date.now(),
            depth: state.currentDepth + 1,
            novelty: state.currentNovelty,
            surprise: state.currentSurprise
        };

        state.trajectory.push(node);
        state.currentNode = node;
        state.currentDepth = node.depth;
        state.totalBranches++;

        const heartBtn = document.getElementById('heart-btn');
        heartBtn.querySelector('.heart-icon').innerHTML = '&#10084;';
        heartBtn.classList.remove('active');

        updateTrajectoryList();
        updateStats();
        updateCuriositySignals();
        drawArchiveVisualization();

        const questions = await generateQuestionsWithCuriosity(question, answer, state.trajectory);
        const readDelay = Math.max(0, 5000 - (Date.now() - answerRevealTimestamp));
        if (readDelay > 0) {
            await wait(readDelay);
        }

        displayQuestionOptions(questions);
        setFollowUpAreaVisible(true);

        const shouldMutate = (
            state.currentNovelty > 0.5 ||
            state.currentSurprise > 0.5 ||
            (state.curiosityOutput.explorationBonus > 0.6 && Math.random() < CONFIG.mutationRate)
        );

        if (shouldMutate) {
            state.curiosityNet.mutate();
            drawNetworkVisualization();
        }

    } catch (error) {
        console.error('Selection error:', error);
        stopContextQuestionIntro();
        alert('Error generating response.\n\nError: ' + error.message);
    }
}

function displayQuestionOptions(questions) {
    const optionsContainer = document.getElementById('question-options');
    optionsContainer.innerHTML = '';

    questions.forEach((q, index) => {
        const option = document.createElement('div');
        option.className = 'question-option';
        option.classList.add('reveal-card');
        option.style.setProperty('--reveal-delay', `${index * 85}ms`);

        option.innerHTML = `
            <span class="question-text">${q}</span>
        `;

        option.onclick = () => selectQuestion(q);
        optionsContainer.appendChild(option);
    });
}

function toggleHeart() {
    if (!state.currentNode) return;

    state.currentNode.hearted = !state.currentNode.hearted;

    if (state.currentNode.hearted) {
        state.heartedCount++;
        state.curiosityNet.mutate();
    } else {
        state.heartedCount--;
    }

    const heartBtn = document.getElementById('heart-btn');
    heartBtn.querySelector('.heart-icon').innerHTML = '&#10084;';
    heartBtn.classList.toggle('active', state.currentNode.hearted);

    updateTrajectoryList();
    updateStats();
    drawNetworkVisualization();
}

function resetExploration() {
    if (state.trajectory.length > 0) {
        if (!confirm('Begin a new quest? Your current journey will be cleared.')) {
            return;
        }
    }

    state.currentDepth = 0;
    state.totalBranches = 0;
    state.heartedCount = 0;
    state.trajectory = [];
    state.currentNode = null;
    state.currentNovelty = 0;
    state.currentSurprise = 0;
    state.informationGain = 0;

    state.noveltyArchive = new NoveltyArchive();
    state.predictionModel = new PredictionModel();

    document.getElementById('start-screen').style.display = 'block';
    document.getElementById('exploration-view').style.display = 'none';
    presentSeedArchiveWithIntro();

    updateStats();
    updateTrajectoryList();
    updateCuriositySignals();
    drawNetworkVisualization();
    drawArchiveVisualization();
}

function exportJourney() {
    const data = {
        trajectory: state.trajectory,
        curiosityNet: state.curiosityNet.toJSON(),
        noveltyArchive: state.noveltyArchive.archive,
        stats: {
            depth: state.currentDepth,
            branches: state.totalBranches,
            hearted: state.heartedCount,
            frontiersFound: state.noveltyArchive.frontiersFound
        },
        exportedAt: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `side-quest-journey-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// ============================================
// EVENT LISTENERS
// ============================================

document.addEventListener('DOMContentLoaded', async () => {
    initializePromptEvolution();
    const connectionCheckPromise = checkConnection();
    const archiveLoadPromise = loadSeedArchive();
    presentSeedArchiveWithIntro(archiveLoadPromise);
    await connectionCheckPromise;

    setInterval(async () => {
        if (!state.connected) {
            await checkConnection();
        }
    }, 30000);

    document.getElementById('seed-options').addEventListener('click', (event) => {
        const option = event.target.closest('.seed-option');
        if (!option) return;
        const seed = option.dataset.seed;
        if (seed) {
            startExploration(seed);
        }
    });

    document.getElementById('seed-submit').onclick = () => {
        const input = document.getElementById('seed-input');
        const question = input.value.trim();
        if (question) {
            startExploration(question);
        }
    };

    document.getElementById('seed-input').onkeypress = (e) => {
        if (e.key === 'Enter') {
            const question = e.target.value.trim();
            if (question) {
                startExploration(question);
            }
        }
    };

    document.getElementById('custom-submit').onclick = () => {
        const input = document.getElementById('custom-input');
        const question = input.value.trim();
        if (question) {
            selectQuestion(question);
            input.value = '';
        }
    };

    document.getElementById('custom-input').onkeypress = (e) => {
        if (e.key === 'Enter') {
            const question = e.target.value.trim();
            if (question) {
                selectQuestion(question);
                e.target.value = '';
            }
        }
    };

    document.getElementById('heart-btn').onclick = toggleHeart;

    document.getElementById('restart-btn').onclick = resetExploration;
    document.getElementById('export-btn').onclick = exportJourney;

    document.getElementById('num-questions').oninput = (e) => {
        CONFIG.numQuestions = parseInt(e.target.value);
        document.getElementById('num-questions-val').textContent = e.target.value;
    };

    document.getElementById('mutation-rate').oninput = (e) => {
        CONFIG.mutationRate = parseFloat(e.target.value);
        document.getElementById('mutation-rate-val').textContent = e.target.value;
    };

    document.getElementById('novelty-threshold').oninput = (e) => {
        CONFIG.noveltyThreshold = parseFloat(e.target.value);
        document.getElementById('novelty-threshold-val').textContent = e.target.value;
    };

    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT') return;

        if (e.key >= '1' && e.key <= '6') {
            const index = parseInt(e.key) - 1;
            const options = document.querySelectorAll('.question-option');
            if (options[index]) {
                options[index].click();
            }
        }

        if (e.key.toLowerCase() === 'h') {
            toggleHeart();
        }
    });

    // Sidebar toggle logic
    const mainContainer = document.getElementById('main-container');
    const collapseLeft = document.getElementById('toggle-left');
    const collapseRight = document.getElementById('toggle-right');
    const reopenLeft = document.getElementById('reopen-left');
    const reopenRight = document.getElementById('reopen-right');

    function updateSidebarState() {
        const leftCollapsed = mainContainer.classList.contains('left-collapsed');
        const rightCollapsed = mainContainer.classList.contains('right-collapsed');
        reopenLeft.style.display = leftCollapsed ? 'inline-block' : 'none';
        reopenRight.style.display = rightCollapsed ? 'inline-block' : 'none';
    }

    function toggleLeftSidebar() {
        mainContainer.classList.toggle('left-collapsed');
        updateSidebarState();
        setTimeout(() => {
            drawNetworkVisualization();
            drawArchiveVisualization();
        }, 320);
    }

    function toggleRightSidebar() {
        mainContainer.classList.toggle('right-collapsed');
        updateSidebarState();
        setTimeout(() => {
            drawNetworkVisualization();
            drawArchiveVisualization();
        }, 320);
    }

    collapseLeft.onclick = toggleLeftSidebar;
    reopenLeft.onclick = toggleLeftSidebar;
    collapseRight.onclick = toggleRightSidebar;
    reopenRight.onclick = toggleRightSidebar;

    drawNetworkVisualization();
    drawArchiveVisualization();
    updateCuriositySignals();
});

window.addEventListener('resize', () => {
    drawNetworkVisualization();
    drawArchiveVisualization();
});
