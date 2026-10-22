document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const messagesList = document.getElementById('messages-list');
    const charCounter = document.getElementById('char-counter');
    const errorBanner = document.getElementById('error-banner');
    const errorContent = document.getElementById('error-content');
    const errorCloseBtn = document.getElementById('error-close-btn');
    const clearChatBtn = document.getElementById('clear-chat-btn');
    const statusText = document.getElementById('status-text');
    const statusIndicator = document.getElementById('system-status-indicator');
    const suggestionChips = document.querySelectorAll('.suggestion-chip');

    let isProcessing = false;

    // Check System Health on Startup
    async function checkHealth() {
        try {
            const res = await fetch('/api/health');
            if (!res.ok) throw new Error('فشل فحص الحالة');
            const data = await res.json();

            const dot = statusIndicator.querySelector('.status-dot');
            dot.className = 'status-dot';

            if (data.status === 'healthy' && data.openai_configured) {
                dot.classList.add('healthy');
                statusText.textContent = `جاهز (${data.chunks_loaded} مقطع مفهرس)`;
            } else if (!data.openai_configured) {
                dot.classList.add('warning');
                statusText.textContent = 'بانتظار تهيئة مفتاح OpenAI في .env';
            } else {
                dot.classList.add('warning');
                statusText.textContent = 'النظام يعمل بشكل جزئي';
            }
        } catch (err) {
            statusText.textContent = 'تعذر الاتصال بالخادم';
        }
    }

    checkHealth();

    // Character Counter
    userInput.addEventListener('input', () => {
        const len = userInput.value.length;
        charCounter.textContent = `${len} / 500`;
    });

    // Error Handling
    function showError(message) {
        errorContent.textContent = message;
        errorBanner.style.display = 'flex';
        errorBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function hideError() {
        errorBanner.style.display = 'none';
        errorContent.textContent = '';
    }

    errorCloseBtn.addEventListener('click', hideError);

    // Safe Text Node Helper
    function createSafeParagraphs(text, container) {
        const paragraphs = text.split('\n');
        for (const p of paragraphs) {
            const trimmed = p.trim();
            if (!trimmed) continue;

            const pEl = document.createElement('p');

            // Format citations like [chunk_XX] safely
            const regex = /\[(chunk_\d+)\]/g;
            let lastIdx = 0;
            let match;

            while ((match = regex.exec(trimmed)) !== null) {
                if (match.index > lastIdx) {
                    pEl.appendChild(document.createTextNode(trimmed.substring(lastIdx, match.index)));
                }
                const chip = document.createElement('span');
                chip.className = 'citation-chip';
                chip.textContent = match[0];
                pEl.appendChild(chip);
                lastIdx = regex.lastIndex;
            }

            if (lastIdx < trimmed.length) {
                pEl.appendChild(document.createTextNode(trimmed.substring(lastIdx)));
            }

            container.appendChild(pEl);
        }
    }

    // Append User Message to UI
    function appendUserMessage(questionText) {
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble user';

        const meta = document.createElement('div');
        meta.className = 'bubble-meta';
        meta.textContent = 'سؤالك:';

        const body = document.createElement('div');
        body.className = 'bubble-body';
        body.textContent = questionText;

        bubble.appendChild(meta);
        bubble.appendChild(body);
        messagesList.appendChild(bubble);
        messagesList.scrollTop = messagesList.scrollHeight;
    }

    // Append Loading State
    function showLoading() {
        const loadingCard = document.createElement('div');
        loadingCard.className = 'loading-card';
        loadingCard.id = 'active-loading-indicator';

        const spinner = document.createElement('div');
        spinner.className = 'loading-spinner';

        const label = document.createElement('span');
        label.textContent = 'جاري البحث في الوثيقة وصياغة الإجابة...';

        loadingCard.appendChild(spinner);
        loadingCard.appendChild(label);
        messagesList.appendChild(loadingCard);
        messagesList.scrollTop = messagesList.scrollHeight;
    }

    function removeLoading() {
        const loading = document.getElementById('active-loading-indicator');
        if (loading) loading.remove();
    }

    // Append Assistant Message with Sources
    function appendAssistantMessage(data) {
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble assistant';

        const meta = document.createElement('div');
        meta.className = 'bubble-meta';
        meta.textContent = 'إجابة المساعد الذكي:';

        const body = document.createElement('div');
        body.className = 'bubble-body';

        // 1. Answer text (safely rendered)
        const answerBox = document.createElement('div');
        answerBox.className = 'answer-text';
        createSafeParagraphs(data.answer, answerBox);
        body.appendChild(answerBox);

        // 2. Sources (if any)
        if (data.sources && data.sources.length > 0) {
            const sourcesBox = document.createElement('div');
            sourcesBox.className = 'sources-box';

            const sourcesTitle = document.createElement('div');
            sourcesTitle.className = 'sources-title';
            sourcesTitle.textContent = `المصادر الداعمة من الوثيقة (${data.sources.length}):`;
            sourcesBox.appendChild(sourcesTitle);

            const sourcesList = document.createElement('div');
            sourcesList.className = 'sources-list';

            data.sources.forEach(src => {
                const srcCard = document.createElement('div');
                srcCard.className = 'source-card';

                const srcHeader = document.createElement('div');
                srcHeader.className = 'source-header';

                const pageBadge = document.createElement('span');
                pageBadge.className = 'page-badge';
                pageBadge.textContent = `صفحة ${src.page_number} في PDF`;

                const chunkTag = document.createElement('span');
                chunkTag.className = 'chunk-tag';
                chunkTag.textContent = src.chunk_id;

                srcHeader.appendChild(pageBadge);
                srcHeader.appendChild(chunkTag);

                const excerptDiv = document.createElement('div');
                excerptDiv.className = 'source-excerpt';
                excerptDiv.textContent = `"${src.excerpt}"`;

                srcCard.appendChild(srcHeader);
                srcCard.appendChild(excerptDiv);
                sourcesList.appendChild(srcCard);
            });

            sourcesBox.appendChild(sourcesList);
            body.appendChild(sourcesBox);
        }

        // 3. Document disclaimer note
        if (data.disclaimer) {
            const disclaimerBox = document.createElement('div');
            disclaimerBox.className = 'answer-disclaimer';
            disclaimerBox.textContent = data.disclaimer;
            body.appendChild(disclaimerBox);
        }

        bubble.appendChild(meta);
        bubble.appendChild(body);
        messagesList.appendChild(bubble);
        messagesList.scrollTop = messagesList.scrollHeight;
    }

    // Send Question to API
    async function handleQuestionSubmit(question) {
        if (!question || isProcessing) return;

        hideError();
        isProcessing = true;
        sendBtn.disabled = true;
        userInput.disabled = true;

        appendUserMessage(question);
        userInput.value = '';
        charCounter.textContent = '0 / 500';

        showLoading();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question })
            });

            const data = await response.json();

            removeLoading();

            if (!response.ok) {
                const errorMsg = data.detail || `خطأ ${response.status}: تعذر معالجة الطلب`;
                showError(errorMsg);
            } else {
                appendAssistantMessage(data);
            }
        } catch (err) {
            removeLoading();
            showError('تعذر الاتصال بالخادم. يرجى التأكد من تشغيل تطبيق FastAPI بنجاح.');
        } finally {
            isProcessing = false;
            sendBtn.disabled = false;
            userInput.disabled = false;
            userInput.focus();
        }
    }

    // Form Submit Event
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (text) {
            handleQuestionSubmit(text);
        }
    });

    // Suggested Questions Chips
    suggestionChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const q = chip.getAttribute('data-question');
            if (q) {
                userInput.value = q;
                charCounter.textContent = `${q.length} / 500`;
                handleQuestionSubmit(q);
            }
        });
    });

    // Clear Chat
    clearChatBtn.addEventListener('click', () => {
        const welcome = document.getElementById('welcome-message');
        messagesList.innerHTML = '';
        if (welcome) {
            messagesList.appendChild(welcome);
        }
        hideError();
    });
});
