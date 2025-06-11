import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [fadingOut, setFadingOut] = useState(false);
  const chatBoxRef = useRef(null);

  const handleSend = async () => {
    // 사용자가 입력한 메시지가 비어있는지 확인
    if (!input.trim()) return;

    // 사용자가 입력한 메시지를 'user' 타입으로 설정
    const userMessage = { type: 'user', text: input };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput(''); // 입력창 초기화

    try {
    // 사용자가 입력한 질문을 FastAPI 백엔드로 POST 요청
      const res = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },  // JSON 형식으로 요청
        body: JSON.stringify({ 
          question: input, 
          history: updatedMessages.filter(msg => msg.type === 'user' || msg.type === 'bot') }),  // 질문을 JSON 형식으로 보냄
      });

      // 백엔드에서 받은 응답 데이터를 JSON 형식으로 변환
      const data = await res.json();

      // 챗봇의 응답을 'bot' 타입으로 설정
      const botMessage = { type: 'bot', text: data.answer };
      setMessages((prev) => [...prev, botMessage]);   // 이전 메시지 배열에 챗봇의 응답을 추가
    } catch (err) {
      // 서버 오류가 발생한 경우, '서버 오류' 메시지를 챗봇의 응답으로 설정
      setMessages((prev) => [
        ...prev,
        { type: 'bot', text: '서버 오류가 발생했습니다.' },
      ]);
    }
  };

  const handleReset = () => {
    setFadingOut(true);
    setTimeout(() => {
      setMessages([]);
      setFadingOut(false);
    }, 500);
  };
  // 메시지가 변경될 때마다 자동으로 스크롤을 아래로 내리기
  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [messages]); // messages가 변경될 때마다 실행 (새로운 대화내역이 추가될때마다)

  return (
    <div className={`chat-container ${messages.length > 0 ? 'has-chat' : ''}`}>
      <h1>공주대학교 챗봇
        <img src='/knungi_text.png' alt='Knungi TextIMG' style={{ width: '80px', marginLeft: '3px'}} />
      </h1>
      {messages.length > 0 && (
        <button onClick={handleReset} className="reset-button">초기화 ↺</button>
      )}
      <div className='chat-box' ref={chatBoxRef} style={{ display: messages.length > 0 ? 'block' : 'none' }}>
        {messages.map((msg, idx) => (
          <div key={idx} className={`message-wrapper ${msg.type} ${fadingOut ? 'fade-out' : ''}`}>
            {msg.type === 'bot' && (
                <img src='/knungi_face.png' alt='챗봇' className='avatar'/>
            )}
            <div className='message'>
              <div className='name'>{msg.type === 'user' ? '나' : '크눙이'}</div>
              <div className='text'>{msg.text}</div>
            </div>
          </div>
        ))}
      </div>

      <div className='input-area'>
        <input
          type='text'
          placeholder='무엇이든 물어보세요!'
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
        />
        <button onClick={handleSend}>전송</button>
      </div>
    </div>
  );
}

export default App;