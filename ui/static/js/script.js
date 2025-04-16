const openUploadBtn = document.getElementById('open-upload-modal');
const uploadModal = document.getElementById('upload-modal');
const cancelUploadBtn = document.getElementById('cancel-upload');
const submitUploadBtn = document.getElementById('submit-upload');
const openAssistantBtn = document.getElementById('open-assistant');
const aiAssistant = document.getElementById('ai-assistant');
const closeAssistantBtn = document.getElementById('close-assistant');
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const browseFilesBtn = document.getElementById('browse-files');
const chatInput = document.getElementById('chat-input');
const sendMessageBtn = document.getElementById('send-message');
const chatContainer = document.getElementById('chat-container');
const transactionList = document.getElementById('transaction-list');
const accountOverview = document.getElementById('account-overview');
const chartsContainer = document.getElementById('charts');
const insightsContainer = document.getElementById('insights');

openUploadBtn.addEventListener('click', () => {
    uploadModal.style.display = 'flex';
});

cancelUploadBtn.addEventListener('click', () => {
    uploadModal.style.display = 'none';
    resetDropArea();
});

openAssistantBtn.addEventListener('click', () => {
    aiAssistant.style.display = 'block';
});

closeAssistantBtn.addEventListener('click', () => {
    aiAssistant.style.display = 'none';
});

browseFilesBtn.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

dropArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropArea.classList.add('drag-over');
});

dropArea.addEventListener('dragleave', () => {
    dropArea.classList.remove('drag-over');
});

dropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dropArea.classList.remove('drag-over');
    handleFiles(e.dataTransfer.files);
});

function handleFiles(files) {
    if (files.length > 10) {
        alert('Maximum 10 files allowed');
        return;
    }
    for (let i = 0; i < files.length; i++) {
        if (!files[i].name.toLowerCase().endsWith('.pdf')) {
            alert('Only PDF files are allowed');
            return;
        }
    }
    dropArea.innerHTML = `<p class="drop-text">${files.length} file(s) selected</p>`;
    if (files.length > 0) {
        const fileList = document.createElement('div');
        fileList.className = 'file-list';
        for (let i = 0; i < files.length; i++) {
            const fileItem = document.createElement('p');
            fileItem.textContent = files[i].name;
            fileItem.className = 'file-item';
            fileList.appendChild(fileItem);
        }
        dropArea.appendChild(fileList);
    }
}

submitUploadBtn.addEventListener('click', async () => {
    const files = fileInput.files;
    if (!files.length) {
        alert('Please select files to upload');
        return;
    }
    submitUploadBtn.disabled = true;
    submitUploadBtn.textContent = 'Processing...';
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }
    try {
        // Parse PDFs
        console.log('Sending request to /parse_pdfs');
        const parseRes = await fetch('/parse_pdfs', {
            method: 'POST',
            body: formData
        });
        console.log('Response status:', parseRes.status);
        if (!parseRes.ok) {
            const errorData = await parseRes.json().catch(() => ({}));
            console.error('Parse PDFs error:', errorData);
            throw new Error(errorData.detail || `Failed to parse PDFs: ${parseRes.statusText}`);
        }
        const parseData = await parseRes.json();
        console.log('Parse PDFs response:', parseData);
        const allTransactionsCsv = parseData.output_csv;

        // Build timeline
        console.log('Sending request to /build_timeline');
        const timelineRes = await fetch('/build_timeline', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input_csv: allTransactionsCsv })
        });
        console.log('Response status:', timelineRes.status);
        if (!timelineRes.ok) {
            const errorData = await timelineRes.json().catch(() => ({}));
            console.error('Build timeline error:', errorData);
            throw new Error(errorData.detail || `Failed to build timeline: ${timelineRes.statusText}`);
        }
        const timelineData = await timelineRes.json();
        console.log('Build timeline response:', timelineData);
        const timelineCsv = timelineData.output_csv;

        // Categorize transactions
        console.log('Sending request to /categorize_transactions');
        const categorizeRes = await fetch('/categorize_transactions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input_csv: timelineCsv })
        });
        console.log('Response status:', categorizeRes.status);
        if (!categorizeRes.ok) {
            const errorData = await categorizeRes.json().catch(() => ({}));
            console.error('Categorize transactions error:', errorData);
            throw new Error(errorData.detail || `Failed to categorize transactions: ${categorizeRes.statusText}`);
        }
        const categorizeData = await categorizeRes.json();
        console.log('Categorize transactions response:', categorizeData);
        const categorizedCsv = categorizeData.output_csv;

        // Run analysis, visualizations, stories
        console.log('Sending requests to /analyze_transactions, /generate_visualizations, /generate_stories');
        const [analysisRes, vizRes, storiesRes] = await Promise.all([
            fetch('/analyze_transactions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ input_csv: categorizedCsv })
            }).then(async res => {
                console.log('Analyze transactions status:', res.status);
                if (!res.ok) {
                    const errorData = await res.json().catch(() => ({}));
                    console.error('Analyze transactions error:', errorData);
                    throw new Error(errorData.detail || `Failed to analyze transactions: ${res.statusText}`);
                }
                return res.json();
            }),
            fetch('/generate_visualizations', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ input_csv: categorizedCsv })
            }).then(async res => {
                console.log('Generate visualizations status:', res.status);
                if (!res.ok) {
                    const errorData = await res.json().catch(() => ({}));
                    console.error('Generate visualizations error:', errorData);
                    throw new Error(errorData.detail || `Failed to generate visualizations: ${res.statusText}`);
                }
                return res.json();
            }),
            fetch('/generate_stories', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ input_csv: categorizedCsv })
            }).then(async res => {
                console.log('Generate stories status:', res.status);
                if (!res.ok) {
                    const errorData = await res.json().catch(() => ({}));
                    console.error('Generate stories error:', errorData);
                    throw new Error(errorData.detail || `Failed to generate stories: ${res.statusText}`);
                }
                return res.json();
            })
        ]);

        console.log('Analysis response:', analysisRes);
        console.log('Visualizations response:', vizRes);
        console.log('Stories response:', storiesRes);

        alert(`Successfully processed ${files.length} file(s)`);
        uploadModal.style.display = 'none';
        resetDropArea();
        await loadAllData();
    } catch (error) {
        console.error('Upload error:', error);
        alert(`Error: ${error.message}`);
    } finally {
        submitUploadBtn.disabled = false;
        submitUploadBtn.textContent = 'Upload';
    }
});

sendMessageBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;
    addMessageToChat(message, 'user-message');
    chatInput.value = '';
    try {
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'assistant-message typing';
        typingIndicator.innerHTML = '<p>...</p>';
        chatContainer.appendChild(typingIndicator);
        const response = await fetch('/process_nlp_queries', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                input_csv: 'data/output/categorized.csv',
                nlp_query: { query: message }
            })
        });
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            console.error('NLP query error:', errorData);
            throw new Error(errorData.detail || `Query failed: ${response.statusText}`);
        }
        const result = await response.json();
        chatContainer.removeChild(typingIndicator);
        addMessageToChat(result.result.response || 'No response', 'assistant-message');
        if (result.result.visualization) {
            updateCharts(result.result.visualization);
        }
    } catch (error) {
        chatContainer.removeChild(document.querySelector('.typing') || document.createElement('div'));
        addMessageToChat(`Sorry, I encountered an error: ${error.message}`, 'assistant-message error');
    }
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function loadAllData() {
    try {
        await Promise.all([
            loadTransactions(),
            loadVisualizations(),
            loadInsights()
        ]);
    } catch (error) {
        console.error('Error loading data:', error);
    }
}

async function loadTransactions() {
    const noData = document.getElementById('no-data-transactions');
    try {
        const response = await fetch('/transactions');
        console.log('Transactions response status:', response.status);
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            console.error('Transactions error:', errorData);
            throw new Error(errorData.detail || 'No transactions available');
        }
        const transactions = await response.json();
        console.log('Transactions data:', transactions);
        transactionList.innerHTML = '';
        noData.style.display = 'none';
        if (!transactions.length) {
            noData.textContent = 'No transactions found.';
            noData.style.display = 'block';
            return;
        }
        transactions.forEach(transaction => {
            const transactionElement = document.createElement('div');
            transactionElement.className = 'transaction-item';
            const amount = transaction['Withdrawal (INR)']
                ? -parseFloat(transaction['Withdrawal (INR)'])
                : parseFloat(transaction['Deposit (INR)']);
            const isNegative = amount < 0;
            const amountClass = isNegative ? 'transaction-amount' : 'transaction-amount income';
            const amountText = isNegative
                ? `-₹${Math.abs(amount).toLocaleString('en-IN')}`
                : `₹${amount.toLocaleString('en-IN')}`;
            let badgeClass = 'category-badge';
            if (transaction.category && transaction.category.includes('Income')) {
                badgeClass = 'badge-green';
            } else if (transaction.category && (transaction.category.includes('Entertainment') || transaction.category.includes('Shopping'))) {
                badgeClass = 'badge-red';
            } else if (transaction.category && (transaction.category.includes('Debt Payment') || transaction.category.includes('Utilities'))) {
                badgeClass = 'badge-blue';
            }
            transactionElement.innerHTML = `
                <div class="transaction-info">
                    <span class="transaction-name">${transaction.Narration || 'Unknown'}</span>
                    <span class="transaction-date">${transaction.parsed_date || 'N/A'}</span>
                </div>
                <span class="${amountClass}">${amountText}</span>
                <div class="transaction-actions">
                    <button class="action-button ${badgeClass}">${transaction.category || 'Uncategorized'}</button>
                    <button class="action-button details-button">Details</button>
                    <button class="action-button categorize-button">Categorize</button>
                </div>
            `;
            transactionList.appendChild(transactionElement);
        });
        addTransactionButtonListeners();
    } catch (error) {
        console.error('Error loading transactions:', error);
        transactionList.innerHTML = '';
        noData.textContent = error.message.includes('No transactions') ? 'Please upload bank statements to view transactions.' : `Error: ${error.message}`;
        noData.style.display = 'block';
    }
}

async function loadVisualizations() {
    const noData = document.getElementById('no-data-charts');
    const lineChartCard = document.querySelector('.line-chart-card');
    const pieChartCard = document.querySelector('.pie-chart-card');
    try {
        const response = await fetch('/generate_visualizations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input_csv: 'data/output/categorized.csv' })
        });
        console.log('Visualizations response status:', response.status);
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            console.error('Visualizations error:', errorData);
            throw new Error(errorData.detail || 'No visualizations available');
        }
        const visualizationData = await response.json();
        console.log('Visualizations data:', visualizationData);
        noData.style.display = 'none';
        lineChartCard.style.display = 'block';
        pieChartCard.style.display = 'block';
        // Default data if empty
        const vizResult = visualizationData.result || {};
        const spendingTrends = vizResult.spending_trends || {
            labels: ['Jan', 'Feb'],
            expenses: [0, 0],
            budget: [0, 0]
        };
        const expenseBreakdown = vizResult.expense_breakdown || {
            categories: ['Other'],
            percentages: [100]
        };
        const accountOverviewData = vizResult.account_overview || {
            total_balance: 0,
            monthly_income: 0,
            monthly_expense: 0,
            balance_percentage: 0,
            income_percentage: 0,
            expense_percentage: 0
        };
        updateLineChart(spendingTrends);
        updatePieChart(expenseBreakdown);
        updateAccountOverview(accountOverviewData);
    } catch (error) {
        console.error('Error loading visualizations:', error);
        noData.textContent = error.message.includes('No visualizations') ? 'Please upload bank statements to view spending trends.' : `Error: ${error.message}`;
        noData.style.display = 'block';
        lineChartCard.style.display = 'none';
        pieChartCard.style.display = 'none';
        accountOverview.innerHTML = '<p class="no-data" id="no-data-overview">Please upload bank statements to view your financial overview.</p>';
    }
}

async function loadInsights() {
    const noData = document.getElementById('no-data-insights');
    try {
        const [analysisRes, storiesRes] = await Promise.all([
            fetch('/analyze_transactions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ input_csv: 'data/output/categorized.csv' })
            }).then(async res => {
                console.log('Analyze transactions status:', res.status);
                if (!res.ok) {
                    const errorData = await res.json().catch(() => ({}));
                    throw new Error(errorData.detail || 'Failed to load analysis');
                }
                return res.json();
            }),
            fetch('/generate_stories', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ input_csv: 'data/output/categorized.csv' })
            }).then(async res => {
                console.log('Generate stories status:', res.status);
                if (!res.ok) {
                    const errorData = await res.json().catch(() => ({}));
                    throw new Error(errorData.detail || 'Failed to load stories');
                }
                return res.json();
            })
        ]);
        const analysis = await analysisRes;
        const stories = await storiesRes;
        console.log('Analysis data:', analysis);
        console.log('Stories data:', stories);
        insightsContainer.innerHTML = '';
        noData.style.display = 'none';
        const analysisResult = analysis.result || {};
        const patterns = analysisResult.patterns || [];
        const recurring = analysisResult.recurring || [];
        const anomalies = analysisResult.anomalies || [];
        if (patterns.length) {
            const patternsDiv = document.createElement('div');
            patternsDiv.className = 'insight-item';
            patternsDiv.innerHTML = `<h4>Spending Patterns</h4><ul>${patterns.map(p => `<li>${p}</li>`).join('')}</ul>`;
            insightsContainer.appendChild(patternsDiv);
        }
        if (recurring.length) {
            const recurringDiv = document.createElement('div');
            recurringDiv.className = 'insight-item';
            recurringDiv.innerHTML = `<h4>Recurring Payments</h4><ul>${recurring.map(r => `<li>${r.narration}: ₹${r.amount} (${r.frequency})</li>`).join('')}</ul>`;
            insightsContainer.appendChild(recurringDiv);
        }
        if (anomalies.length) {
            const anomaliesDiv = document.createElement('div');
            anomaliesDiv.className = 'insight-item';
            anomaliesDiv.innerHTML = `<h4>Anomalies</h4><ul>${anomalies.map(a => `<li>${a.Narration}: ₹${a.amount} (${a.severity})</li>`).join('')}</ul>`;
            insightsContainer.appendChild(anomaliesDiv);
        }
        if (stories.result && stories.result.length) {
            const storiesDiv = document.createElement('div');
            storiesDiv.className = 'insight-item';
            storiesDiv.innerHTML = `<h4>Monthly Insights</h4><ul>${stories.result.map(s => `<li>${s}</li>`).join('')}</ul>`;
            insightsContainer.appendChild(storiesDiv);
        }
        if (!insightsContainer.children.length) {
            noData.textContent = 'No insights available.';
            noData.style.display = 'block';
        }
    } catch (error) {
        console.error('Error loading insights:', error);
        insightsContainer.innerHTML = '';
        noData.textContent = error.message.includes('No analysis') ? 'Please upload bank statements to view financial insights.' : `Error: ${error.message}`;
        noData.style.display = 'block';
    }
}

function updateAccountOverview(overview) {
    accountOverview.innerHTML = `
        <div class="card">
            <div class="card-content">
                <p class="card-label">Total Balance</p>
                <p class="card-value">₹${formatNumber(overview.total_balance)}</p>
                <span class="percentage-badge ${overview.balance_percentage >= 0 ? 'badge-blue' : 'badge-red'}">${overview.balance_percentage >= 0 ? '+' : ''}${overview.balance_percentage}%</span>
                <p class="card-subtitle">From last month</p>
            </div>
        </div>
        <div class="card card-income">
            <div class="card-content">
                <p class="card-label">Monthly Income</p>
                <p class="card-value">₹${formatNumber(overview.monthly_income)}</p>
                <span class="percentage-badge ${overview.income_percentage >= 0 ? 'badge-green' : 'badge-red'}">${overview.income_percentage >= 0 ? '+' : ''}${overview.income_percentage}%</span>
                <p class="card-subtitle">From last month</p>
            </div>
        </div>
        <div class="card card-expense">
            <div class="card-content">
                <p class="card-label">Monthly Expense</p>
                <p class="card-value">₹${formatNumber(overview.monthly_expense)}</p>
                <span class="percentage-badge ${overview.expense_percentage <= 0 ? 'badge-green' : 'badge-red'}">${overview.expense_percentage >= 0 ? '+' : ''}${overview.expense_percentage}%</span>
                <p class="card-subtitle">From last month</p>
            </div>
        </div>
    `;
}

function updateLineChart(trendData) {
    const ctx = document.getElementById('line-chart').getContext('2d');
    if (window.lineChart) {
        window.lineChart.destroy();
    }
    window.lineChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: trendData.labels || ['No Data'],
            datasets: [
                {
                    label: 'Expenses',
                    data: trendData.expenses || [0],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.3,
                    fill: true
                },
                {
                    label: 'Budget',
                    data: trendData.budget || [0],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    borderDash: [5, 5],
                    tension: 0.3,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Spending Trends' }
            },
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}

function updatePieChart(breakdownData) {
    const ctx = document.getElementById('pie-chart').getContext('2d');
    if (window.pieChart) {
        window.pieChart.destroy();
    }
    const legendsContainer = document.querySelector('.chart-legends');
    legendsContainer.innerHTML = '';
    const categories = breakdownData.categories || ['No Data'];
    const percentages = breakdownData.percentages || [100];
    categories.forEach((category, index) => {
        const percentage = percentages[index] || 0;
        const legendItem = document.createElement('div');
        legendItem.className = 'legend-item';
        legendItem.innerHTML = `
            <div class="legend-color" style="background-color: ${['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'][index % 5]};"></div>
            <span class="legend-label">${category} (${percentage}%)</span>
        `;
        legendsContainer.appendChild(legendItem);
    });
    window.pieChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: categories,
            datasets: [{
                data: percentages,
                backgroundColor: ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'],
                borderWidth: 0,
                borderRadius: 5
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                title: { display: true, text: 'Expense Breakdown' }
            }
        }
    });
}

function addMessageToChat(message, className) {
    const messageDiv = document.createElement('div');
    messageDiv.className = className;
    messageDiv.innerHTML = `<p>${message}</p>`;
    chatContainer.appendChild(messageDiv);
}

function resetDropArea() {
    dropArea.innerHTML = `
        <p class="drop-text">Drag & Drop PDF Files Here</p>
        <p class="or-text">or</p>
        <button class="browse-button" id="browse-files">Browse Files</button>
    `;
    document.getElementById('browse-files').addEventListener('click', () => {
        fileInput.click();
    });
    fileInput.value = '';
}

function addTransactionButtonListeners() {
    document.querySelectorAll('.details-button').forEach(button => {
        button.addEventListener('click', function() {
            const transactionName = this.parentElement.parentElement.querySelector('.transaction-name').textContent;
            alert(`Details for ${transactionName}`);
        });
    });
    document.querySelectorAll('.categorize-button').forEach(button => {
        button.addEventListener('click', function() {
            const transactionName = this.parentElement.parentElement.querySelector('.transaction-name').textContent;
            const currentCategory = this.parentElement.querySelector('.category-badge, .badge-green, .badge-red, .badge-blue').textContent;
            const newCategory = prompt(`Current category: ${currentCategory}\nEnter new category for ${transactionName}:`, currentCategory);
            if (newCategory && newCategory !== currentCategory) {
                this.parentElement.querySelector('.category-badge, .badge-green, .badge-red, .badge-blue').textContent = newCategory;
                alert(`Category updated to ${newCategory}`);
            }
        });
    });
}

function formatNumber(number) {
    return Number(number).toLocaleString('en-IN');
}

window.addEventListener('load', () => {
    addMessageToChat("Hello! I'm your AI financial assistant. Please upload bank statements to get started.", 'assistant-message');
    loadAllData();
});