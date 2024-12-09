import { initializeApp } from "https://www.gstatic.com/firebasejs/10.13.1/firebase-app.js";
import { getStorage, ref, uploadBytes, getDownloadURL } from "https://www.gstatic.com/firebasejs/10.13.1/firebase-storage.js";
import { getAuth, signInWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/10.13.1/firebase-auth.js";

// Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyCmIOrT1sVMTuR3kC_UF2VikkVa2Ct7gq8",
    authDomain: "tm-project-c29c8.firebaseapp.com",
    databaseURL: "https://tm-project-c29c8-default-rtdb.asia-southeast1.firebasedatabase.app",
    projectId: "tm-project-c29c8",
    storageBucket: "tm-project-c29c8.appspot.com",
    messagingSenderId: "85473383713",
    appId: "1:85473383713:web:811961152070add704da60",
    measurementId: "G-RWGPQZCQZF"
  };

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const storage = getStorage(app);
const auth = getAuth(app);  // Initialize authentication

let conversations = [];
let currentConversationIndex = null;
// 取得使用者 ID
const userId = localStorage.getItem('userId');
if (!userId) {
    alert("使用者未登入，請先登入");
    window.location.href = 'login.html';
}

document.getElementById('newConversationBtn').addEventListener('click', startNewConversation);

function handleEnter(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

window.handleEnter = handleEnter;  // 將 handleEnter 函數綁定到 window

document.getElementById('switchUserBtn').addEventListener('click', function() {
    // 清除當前的使用者登入狀態（如果需要）
    auth.signOut().then(() => {
        console.log('User signed out.');
        // 跳轉到登入頁面
        window.location.href = 'login.html';
    }).catch((error) => {
        console.error('Sign-out error:', error);
    });
});


// 更新 sendMessage 函數來調用新的 API
function sendMessage() {
    const userInput = document.getElementById('userInput');
    const message = userInput.value.trim();

    if (message) {
        if (currentConversationIndex === null) {
            const defaultName = `Conversation ${conversations.length + 1}`;
            const conversation = { name: defaultName, id: `${userId}_conv_${Date.now()}`, messages: [] }; // 確保有 id
            conversations.push(conversation);
            currentConversationIndex = conversations.length - 1;
            updateConversationTitle();
            renderConversations();
        }

        const chatContent = document.getElementById('chat-content');
        const conversationId = conversations[currentConversationIndex].id; // 獲取當前 conversation 的 id
        
        const userMessage = document.createElement('div');
        userMessage.classList.add('message', 'user');
        userMessage.textContent = message;
        chatContent.appendChild(userMessage);

        const timestamp = new Date().toLocaleString();
        conversations[currentConversationIndex].messages.push({ sender: 'user', text: message, time: timestamp });

        userInput.value = '';
        chatContent.scrollTop = chatContent.scrollHeight;

        // 發送請求到 Flask 後端，並包含 conversationId
        fetch('http://10.1.10.30:5000/api/get-response', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: message, conversationId: conversationId })  // 傳遞 conversationId
        })
        .then(response => response.json())
        .then(data => {
            if (data.answer) {  
                const botMessage = document.createElement('div');
                botMessage.classList.add('message', 'bot');
                botMessage.innerHTML = data.answer.replace(/\\n/g, "<br>");

                const botFeedbackContainer = document.createElement('div');
                botFeedbackContainer.classList.add('feedback-container');

                const botLikeBtn = document.createElement('button');
                botLikeBtn.textContent = '👍🏻';
                botLikeBtn.classList.add('like-btn');
                botLikeBtn.addEventListener('click', () => {
                    const messageIndex = conversations[currentConversationIndex].messages.length - 1;
                    recordFeedback(currentConversationIndex, messageIndex, 1);
                });

                const botDislikeBtn = document.createElement('button');
                botDislikeBtn.textContent = '👎🏻';
                botDislikeBtn.classList.add('dislike-btn');
                botDislikeBtn.addEventListener('click', () => {
                    const messageIndex = conversations[currentConversationIndex].messages.length - 1;
                    recordFeedback(currentConversationIndex, messageIndex, -1);
                });

                botFeedbackContainer.appendChild(botLikeBtn);
                botFeedbackContainer.appendChild(botDislikeBtn);
                botMessage.appendChild(botFeedbackContainer);

                chatContent.appendChild(botMessage);

                const timestamp = new Date().toLocaleString();
                conversations[currentConversationIndex].messages.push({ sender: 'bot', text: data.answer, time: timestamp, feedback: 0 });

                chatContent.scrollTop = chatContent.scrollHeight;
                saveConversationToFile();
            } else {
                console.error('AI 回應錯誤:', data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
}


function recordFeedback(conversationIndex, messageIndex, feedback) {
    const conversation = conversations[conversationIndex];
    if (conversation && conversation.messages[messageIndex] && conversation.messages[messageIndex].sender === 'bot') {
        conversation.messages[messageIndex].feedback = feedback;
        saveConversationToFile();
    } else {
        console.error('無法找到指定的 AI 訊息或訊息索引不正確');
    }
}




function startNewConversation() {
    let conversationName;
    do {
        conversationName = prompt('請輸入對話名稱', `Conversation ${conversations.length + 1}`);
    } while (conversations.some(conv => conv.name === conversationName));

    if (conversationName) {
        const conversation = { name: conversationName, id: `${userId}_conv_${Date.now()}`, messages: [] }; // 添加唯一的 id
        conversations.push(conversation);
        currentConversationIndex = conversations.length - 1;

        updateConversationTitle();
        renderConversations();
        document.getElementById('chat-content').innerHTML = '';
    }
}


function renderConversations() {
    const conversationList = document.getElementById('conversationList');
    conversationList.innerHTML = '';

    conversations.forEach((conversation, index) => {
        const conversationItem = document.createElement('div');
        conversationItem.classList.add('conversation-item');

        const nameDiv = document.createElement('span');
        nameDiv.textContent = conversation.name;
        conversationItem.appendChild(nameDiv);

        const actionsDiv = document.createElement('div');
        actionsDiv.classList.add('actions');

        const editBtn = document.createElement('button');
        editBtn.classList.add('icon-btn');
        editBtn.innerHTML = '✏️';
        editBtn.addEventListener('click', () => renameConversation(index));
        actionsDiv.appendChild(editBtn);

        const deleteBtn = document.createElement('button');
        deleteBtn.classList.add('icon-btn');
        deleteBtn.innerHTML = '🗑️';
        deleteBtn.addEventListener('click', () => deleteConversation(index));
        actionsDiv.appendChild(deleteBtn);

        conversationItem.appendChild(actionsDiv);
        conversationItem.addEventListener('click', () => loadConversation(index));
        conversationList.appendChild(conversationItem);
    });

    conversationList.scrollTop = conversationList.scrollHeight;
}

function loadConversation(index) {
    currentConversationIndex = index;
    const chatContent = document.getElementById('chat-content');
    chatContent.innerHTML = '';  // 清空現有的內容

    const conversation = conversations[index];
    conversation.messages.forEach((message, messageIndex) => {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', message.sender);

        // 設定訊息內容，將換行符號替換為 <br> 標籤
        const messageText = document.createElement('span');
        messageText.innerHTML = `${message.text.replace(/\\n/g, "<br>")} `;
        messageDiv.appendChild(messageText);

        // 如果是 AI 回覆，則顯示讚和倒讚按鈕
        if (message.sender === 'bot') {
            const feedbackContainer = document.createElement('div');
            feedbackContainer.classList.add('feedback-container');

            const likeBtn = document.createElement('button');
            likeBtn.textContent = '👍🏻';
            likeBtn.classList.add('like-btn');
            likeBtn.addEventListener('click', () => recordFeedback(index, messageIndex, 1));

            const dislikeBtn = document.createElement('button');
            dislikeBtn.textContent = '👎🏻';
            dislikeBtn.classList.add('dislike-btn');
            dislikeBtn.addEventListener('click', () => recordFeedback(index, messageIndex, -1));

            feedbackContainer.appendChild(likeBtn);
            feedbackContainer.appendChild(dislikeBtn);
            messageDiv.appendChild(feedbackContainer);
        }

        chatContent.appendChild(messageDiv);
    });

    chatContent.scrollTop = chatContent.scrollHeight;
    updateConversationTitle();
}



function updateConversationTitle() {
    if (currentConversationIndex !== null) {
        const conversation = conversations[currentConversationIndex];
        document.getElementById('conversationTitle').textContent = `TM ROBOT Instructor（${conversation.name}）`;
    } else {
        document.getElementById('conversationTitle').textContent = 'TM ROBOT Instructor';
    }
}

function renameConversation(index) {
    const newName = prompt('輸入新對話名稱', conversations[index].name);
    if (newName) {
        conversations[index].name = newName;
        updateConversationTitle();
        renderConversations();
    }
}

function deleteConversation(index) {
    if (confirm('確定要刪除這個對話嗎？')) {
        conversations.splice(index, 1);
        currentConversationIndex = null;
        updateConversationTitle();
        renderConversations();
        document.getElementById('chat-content').innerHTML = '';
    }
}

function saveConversationToFile() {
    if (currentConversationIndex !== null && auth.currentUser) {  // 確保用戶已登入
        const conversation = conversations[currentConversationIndex];
        let content = '';

        conversation.messages.forEach(message => {
            content += `${message.sender === 'user' ? '使用者' : 'AI'}: ${message.text} (${message.time})\n`;
            
            // 只在 AI 回覆時記錄回饋
            if (message.sender === 'bot') {
                content += `回饋：${message.feedback === 1 ? '1' : message.feedback === -1 ? '-1' : '0'}\n`;
            }
        });
        

        //console.log("儲存的對話內容:", content);  // 確認回饋是否正確顯示

        const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });

        // 使用者 ID
        const userId = auth.currentUser.uid;

        // 檔案名稱：對話名稱 + 用戶 ID
        const fileName = `${userId}_${conversation.name}.txt`;

        const storageRef = ref(storage, fileName);

        uploadBytes(storageRef, blob)
            .then((snapshot) => {
                //console.log('上傳完成:', snapshot);
                return getDownloadURL(snapshot.ref);
            })
            .catch((error) => {
                //console.error('上傳失敗:', error);
            });
    } else {
        console.warn("無法儲存對話，未選擇對話或用戶未登入。");
    }
}


renderConversations();
