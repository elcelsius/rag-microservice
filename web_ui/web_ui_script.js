// web_ui/script.js
document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = userInput.value.trim();
        if (!question) return;

        // Adiciona a mensagem do usuário à tela
        appendMessage(question, 'user');
        userInput.value = '';

        // Mostra o indicador de "digitando"
        const typingIndicator = showTypingIndicator();

        try {
            // Faz a chamada para a nossa API backend
            const response = await fetch('http://localhost:5000/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question: question })
            });

            // Remove o indicador de "digitando"
            typingIndicator.remove();

            if (!response.ok) {
                throw new Error(`Erro na API: ${response.statusText}`);
            }

            const data = await response.json();
            // Adiciona a resposta da IA à tela
            appendMessage(data.answer, 'ai');

        } catch (error) {
            console.error('Falha ao buscar resposta:', error);
            appendMessage('Desculpe, não consegui me conectar ao meu cérebro. Tente novamente mais tarde.', 'ai');
        }
    });

    function appendMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);

        const avatarDiv = document.createElement('div');
        avatarDiv.classList.add('avatar');
        avatarDiv.textContent = sender === 'user' ? 'Você' : 'IA';

        const textDiv = document.createElement('div');
        textDiv.classList.add('text');
        textDiv.textContent = text;

        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(textDiv);
        chatBox.appendChild(messageDiv);

        // Rola para a mensagem mais recente
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