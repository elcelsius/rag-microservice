// web_ui/script.js
document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = userInput.value.trim();
        if (!question) return;

        appendUserMessage(question);
        userInput.value = '';

        const typingIndicator = showTypingIndicator();

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question: question })
            });

            typingIndicator.remove();

            if (!response.ok) {
                throw new Error(`Erro na API: ${response.statusText}`);
            }

            const data = await response.json();
            appendAiMessage(data); // Passa o objeto de dados completo

        } catch (error) {
            console.error('Falha ao buscar resposta:', error);
            appendAiMessage({ answer: 'Desculpe, não consegui me conectar ao meu cérebro. Tente novamente mais tarde.' });
        }
    });

    function appendUserMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'user-message');

        messageDiv.innerHTML = `
            <div class="text">${text}</div>
            <div class="avatar">Você</div>
        `;
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function appendAiMessage(data) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'ai-message');

        const textDiv = document.createElement('div');
        textDiv.classList.add('text');
        textDiv.textContent = data.answer;

        // Lógica para adicionar citações
        if (data.citations && data.citations.length > 0) {
            const citationsContainer = document.createElement('div');
            citationsContainer.classList.add('citations');

            const title = document.createElement('h4');
            title.textContent = 'Fontes Consultadas:';
            citationsContainer.appendChild(title);

            data.citations.forEach(citation => {
                const citationDiv = document.createElement('div');
                citationDiv.classList.add('citation');
                citationDiv.innerHTML = `
                    <p class="citation-source">${citation.documento}</p>
                    <p class="citation-snippet">"...${citation.trecho}..."</p>
                `;
                citationsContainer.appendChild(citationDiv);
            });
            textDiv.appendChild(citationsContainer);
        }

        const avatarDiv = document.createElement('div');
        avatarDiv.classList.add('avatar');
        avatarDiv.textContent = 'IA';

        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(textDiv);
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.classList.add('message', 'ai-message', 'typing-indicator');
        indicator.innerHTML = `
            <div class="avatar">IA</div>
            <div class="text">Pensando...</div>
        `;
        chatBox.appendChild(indicator);
        chatBox.scrollTop = chatBox.scrollHeight;
        return indicator;
    }
});