const chatContainer = document.getElementById('chatContainer');
const chatForm = document.getElementById('chatForm');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const valLang = document.getElementById('valLang');
const valSentiment = document.getElementById('valSentiment');
const valIntent = document.getElementById('valIntent');
const valEscalation = document.getElementById('valEscalation');
const retrievalContainer = document.getElementById('retrievalContainer');

function appendMessage(sender, text, metadata = null) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}-msg`;

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.textContent = text;
    msgDiv.appendChild(bubble);

    if (metadata) {
        const badges = document.createElement('div');
        badges.className = 'msg-badges';

        if (metadata.language) {
            const langPill = document.createElement('span');
            langPill.className = 'meta-pill pill-lang';
            langPill.textContent = `${metadata.language.language_name} (${Math.round(metadata.language.confidence * 100)}%)`;
            badges.appendChild(langPill);
        }

        if (metadata.sentiment) {
            const sentPill = document.createElement('span');
            const sent = metadata.sentiment.sentiment;
            const sentClass = sent === 'positive' ? 'pill-pos' : (sent === 'negative' ? 'pill-neg' : 'pill-neu');
            sentPill.className = `meta-pill ${sentClass}`;
            sentPill.textContent = `${sent} (${Math.round(metadata.sentiment.confidence * 100)}%)`;
            badges.appendChild(sentPill);
        }

        if (metadata.intent) {
            const intentPill = document.createElement('span');
            intentPill.className = 'meta-pill pill-intent';
            intentPill.textContent = metadata.intent.intent.replace(/_/g, ' ');
            badges.appendChild(intentPill);
        }

        if (metadata.escalation_required) {
            const escPill = document.createElement('span');
            escPill.className = 'meta-pill pill-esc';
            escPill.textContent = metadata.escalation_ticket_id || 'ESCALATED';
            badges.appendChild(escPill);
        }

        msgDiv.appendChild(badges);
    }

    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function updatePipelineInspector(data) {
    if (data.language) {
        valLang.textContent = `${data.language.language_name} [${data.language.language}] (${Math.round(data.language.confidence * 100)}%)`;
    }
    if (data.sentiment) {
        valSentiment.textContent = `${data.sentiment.sentiment.toUpperCase()} (${Math.round(data.sentiment.confidence * 100)}%)`;
    }
    if (data.intent) {
        valIntent.textContent = `${data.intent.intent} (${Math.round(data.intent.confidence * 100)}%)`;
    }
    if (data.escalation_required) {
        valEscalation.textContent = `Priority Flagged: ${data.escalation_ticket_id}`;
        valEscalation.style.color = '#f59e0b';
    } else {
        valEscalation.textContent = 'Standard Route (No Escalation)';
        valEscalation.style.color = '#10b981';
    }

    retrievalContainer.innerHTML = '';
    if (data.retrieved_context && data.retrieved_context.length > 0) {
        data.retrieved_context.forEach((chunk, idx) => {
            const card = document.createElement('div');
            card.className = 'chunk-card';

            const header = document.createElement('div');
            header.className = 'chunk-header';

            const badge = document.createElement('span');
            badge.className = 'chunk-badge';
            badge.textContent = `Chunk #${idx + 1} &bull; ${chunk.category || 'SUPPORT'}`;
            header.appendChild(badge);
            card.appendChild(header);

            const inst = document.createElement('div');
            inst.className = 'chunk-instruction';
            inst.textContent = `Q: ${chunk.instruction}`;
            card.appendChild(inst);

            const resp = document.createElement('div');
            resp.className = 'chunk-response';
            resp.textContent = `A: ${chunk.response}`;
            card.appendChild(resp);

            retrievalContainer.appendChild(card);
        });
    } else {
        const empty = document.createElement('div');
        empty.className = 'empty-state';
        empty.textContent = 'No external retrieval needed for this message category (handled directly).';
        retrievalContainer.appendChild(empty);
    }
}

async function sendMessage(text) {
    if (!text || !text.trim()) return;

    appendMessage('user', text);
    userInput.value = '';
    userInput.disabled = true;
    sendBtn.disabled = true;

    valLang.textContent = 'Processing...';
    valSentiment.textContent = 'Evaluating...';
    valIntent.textContent = 'Routing...';
    valEscalation.textContent = 'Evaluating...';

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();
        appendMessage('bot', data.response, data);
        updatePipelineInspector(data);
    } catch (err) {
        appendMessage('bot', 'A connection error occurred while communicating with the support pipeline.');
    } finally {
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
}

chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    sendMessage(userInput.value);
});

document.querySelectorAll('.sample-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
        const query = btn.getAttribute('data-query');
        if (query) {
            userInput.value = query;
            sendMessage(query);
        }
    });
});
