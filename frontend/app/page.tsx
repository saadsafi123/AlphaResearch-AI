/**
 * Frontend UI: AlphaResearch AI
 * Description: A Next.js chat interface for a LangGraph multi-agent backend.
 * Features: Word-by-word streaming simulation, dynamic agent-state tracking, 
 * and Human-in-the-Loop (HITL) interrupt handling.
 */

"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Send, Loader2, AlertCircle, Bot, User, Sparkles, Database, ShieldCheck, PenTool, MessageSquare } from "lucide-react";

// --- CUSTOM HOOK: Word-by-Word Typewriter Effect ---
// Ensures pure rendering by syncing text state directly during the render phase.
function useTypewriter(text: string, speed: number = 10) {
  const [displayedText, setDisplayedText] = useState("");
  const [prevText, setPrevText] = useState(text);

  if (text !== prevText) {
    setPrevText(text);
    setDisplayedText("");
  }

  useEffect(() => {
    if (!text) return;

    let currentIdx = 0;
    const interval = setInterval(() => {
      setDisplayedText(text.substring(0, currentIdx));
      currentIdx++;
      if (currentIdx > text.length) clearInterval(interval);
    }, speed);

    return () => clearInterval(interval);
  }, [text, speed]);

  return displayedText;
}

// --- SUB-COMPONENT: AI Message with Typewriter ---
// Safely handles undefined text lengths and renders Markdown with GitHub Flavored Markdown (tables, lists).
const AiMessage = ({ content = "" }: { content: string }) => {
  const typedContent = useTypewriter(content, 10);
  
  const typedLength = typedContent?.length || 0;
  const totalLength = content?.length || 0;
  
  return (
    <div className="prose prose-slate max-w-none prose-headings:font-semibold prose-a:text-blue-600">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {typedContent}
      </ReactMarkdown>
      {totalLength > 0 && typedLength < totalLength && (
        <span className="inline-block w-2 h-4 bg-blue-500 ml-1 animate-pulse" />
      )}
    </div>
  );
};

// --- MAIN COMPONENT ---
type Message = { role: "user" | "ai"; content: string };

export default function ChatPage() {
  // State Management
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [needsClarification, setNeedsClarification] = useState(false);
  const [activeAgent, setActiveAgent] = useState(0); 

  // Conversation Thread ID (Initialized safely to prevent impure renders)
  const [threadId, setThreadId] = useState(() => Math.random().toString(36).substring(7));
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to the latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // Simulate Agent Pipeline tracking while waiting for the backend
  useEffect(() => {
    if (!isLoading) return;
    
    const timers = [
      setTimeout(() => setActiveAgent(1), 1500),  // Clarity -> Research
      setTimeout(() => setActiveAgent(2), 5000),  // Research -> Validator
      setTimeout(() => setActiveAgent(3), 8000),  // Validator -> Synthesis
    ];
    return () => timers.forEach(clearTimeout);
  }, [isLoading]);

  // Hard resets the chat and generates a new thread ID to clear backend memory
  const resetChat = () => {
    setMessages([]); 
    setInput("");
    setIsLoading(false);
    setNeedsClarification(false);
    setActiveAgent(0);
    setThreadId(Math.random().toString(36).substring(7));
  };

  // Handles API communication with the FastAPI / LangGraph backend
  const sendMessage = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setIsLoading(true);
    setNeedsClarification(false);
    setActiveAgent(0);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage, thread_id: threadId }),
      });

      const data = await response.json();
      setMessages((prev) => [...prev, { role: "ai", content: data.response }]);

      // Check if LangGraph triggered a Human-in-the-Loop circuit breaker
      if (data.status === "interrupt") {
        setNeedsClarification(true);
      }
    } catch (error) {
      setMessages((prev) => [...prev, { role: "ai", content: "Error: Could not reach the research agents. Please ensure the backend is running." }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-white text-slate-800 font-sans">
      
      {/* SIDEBAR (Desktop only) */}
      <div className="hidden md:flex flex-col w-64 bg-slate-50 border-r border-slate-200 p-4">
        <div className="flex items-center space-x-2 mb-8 px-2">
          <div className="bg-blue-600 p-2 rounded-lg text-white">
            <Sparkles className="w-5 h-5" />
          </div>
          <span className="font-bold text-lg text-slate-800">AlphaResearch AI</span>
        </div>
        
        <button 
          onClick={resetChat} 
          className="flex items-center space-x-2 w-full px-4 py-3 bg-white border border-slate-200 rounded-xl hover:bg-slate-50 transition shadow-sm text-sm font-medium text-slate-700"
        >
          <MessageSquare className="w-4 h-4" />
          <span>New Research</span>
        </button>

        <div className="mt-8 px-2">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">Architecture</p>
          <div className="space-y-3 text-sm text-slate-600">
            <div className="flex items-center space-x-2"><Sparkles className="w-4 h-4 text-amber-500"/> <span>Clarity Gatekeeper</span></div>
            <div className="flex items-center space-x-2"><Database className="w-4 h-4 text-blue-500"/> <span>Tavily Researcher</span></div>
            <div className="flex items-center space-x-2"><ShieldCheck className="w-4 h-4 text-green-500"/> <span>Fact Validator</span></div>
            <div className="flex items-center space-x-2"><PenTool className="w-4 h-4 text-purple-500"/> <span>Synthesis Engine</span></div>
          </div>
        </div>
      </div>

      {/* MAIN CHAT AREA */}
      <div className="flex-1 flex flex-col h-full relative">
        {/* Header Mobile */}
        <header className="md:hidden bg-white border-b border-slate-200 p-4 flex items-center justify-center shadow-sm z-10">
          <Sparkles className="w-5 h-5 text-blue-600 mr-2" />
          <h1 className="font-bold text-slate-800">AlphaResearch AI</h1>
        </header>

        {/* Chat History */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 scroll-smooth">
          <div className="max-w-3xl mx-auto space-y-8 pb-32">
            
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-[60vh] text-center space-y-4 animate-in fade-in duration-700">
                <div className="bg-blue-50 p-4 rounded-full">
                  <Sparkles className="w-8 h-8 text-blue-600" />
                </div>
                <h2 className="text-2xl font-bold text-slate-800">What would you like to research?</h2>
                <p className="text-slate-500 max-w-md">
                  I am a multi-agent system. I will analyze your request, search the web, validate facts, and write a comprehensive report.
                </p>
              </div>
            )}

            {messages.map((msg, index) => (
              <div key={index} className="flex space-x-4 animate-in slide-in-from-bottom-4 duration-500">
                {/* Avatar */}
                <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-1 ${
                  msg.role === "user" ? "bg-slate-200 text-slate-600" : "bg-blue-600 text-white shadow-md"
                }`}>
                  {msg.role === "user" ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
                </div>

                {/* Message Content */}
                <div className="flex-1">
                  <div className="font-semibold text-sm mb-1 text-slate-800">
                    {msg.role === "user" ? "You" : "AlphaResearch AI"}
                  </div>
                  <div className="text-slate-700 leading-relaxed">
                    {msg.role === "user" ? (
                      <p>{msg.content}</p>
                    ) : (
                      // Only apply typewriter to the LAST message if it's currently rendering
                      index === messages.length - 1 ? (
                        <AiMessage content={msg.content} />
                      ) : (
                        <div className="prose prose-slate max-w-none prose-headings:font-semibold prose-a:text-blue-600">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                        </div>
                      )
                    )}
                  </div>
                </div>
              </div>
            ))}

            {/* Agent "Thinking" Tracker (Shows while loading) */}
            {isLoading && (
              <div className="flex space-x-4 animate-in fade-in duration-300">
                <div className="w-8 h-8 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center shrink-0 mt-1">
                  <Loader2 className="w-4 h-4 animate-spin" />
                </div>
                <div className="flex-1 bg-slate-50 border border-slate-100 rounded-xl p-4 shadow-sm">
                  <p className="text-sm font-semibold text-slate-700 mb-3 flex items-center">
                    <Sparkles className="w-4 h-4 mr-2 text-blue-600"/> Agent network active...
                  </p>
                  <div className="space-y-2 text-sm">
                    <div className={`flex items-center transition-opacity duration-300 ${activeAgent >= 0 ? "text-slate-700" : "text-slate-400 opacity-50"}`}>
                      {activeAgent > 0 ? <ShieldCheck className="w-4 h-4 mr-2 text-green-500" /> : <Loader2 className="w-4 h-4 mr-2 animate-spin text-blue-500" />}
                      Clarity Agent: Evaluating query intent
                    </div>
                    <div className={`flex items-center transition-opacity duration-300 ${activeAgent >= 1 ? "text-slate-700" : "text-slate-400 opacity-50"}`}>
                      {activeAgent > 1 ? <ShieldCheck className="w-4 h-4 mr-2 text-green-500" /> : activeAgent === 1 ? <Loader2 className="w-4 h-4 mr-2 animate-spin text-blue-500" /> : <span className="w-4 h-4 mr-2" />}
                      Research Agent: Gathering live data (Tavily)
                    </div>
                    <div className={`flex items-center transition-opacity duration-300 ${activeAgent >= 2 ? "text-slate-700" : "text-slate-400 opacity-50"}`}>
                      {activeAgent > 2 ? <ShieldCheck className="w-4 h-4 mr-2 text-green-500" /> : activeAgent === 2 ? <Loader2 className="w-4 h-4 mr-2 animate-spin text-blue-500" /> : <span className="w-4 h-4 mr-2" />}
                      Validator Agent: Auditing & Fact-checking
                    </div>
                    <div className={`flex items-center transition-opacity duration-300 ${activeAgent >= 3 ? "text-slate-700" : "text-slate-400 opacity-50"}`}>
                      {activeAgent === 3 ? <Loader2 className="w-4 h-4 mr-2 animate-spin text-blue-500" /> : <span className="w-4 h-4 mr-2" />}
                      Synthesis Agent: Drafting final report
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} className="h-4" />
          </div>
        </div>

        {/* Input Area (Pinned to bottom) */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-white via-white to-transparent pt-10 pb-6 px-4 md:px-8">
          <div className="max-w-3xl mx-auto">
            {needsClarification && (
              <div className="mb-3 p-3 bg-amber-50 border border-amber-200 text-amber-800 rounded-xl flex items-center shadow-sm animate-in slide-in-from-bottom-2">
                <AlertCircle className="w-5 h-5 mr-2 shrink-0 text-amber-600" />
                <span className="text-sm font-medium">
                  The Clarity Agent needs more context. Please specify below.
                </span>
              </div>
            )}
            
            <form onSubmit={sendMessage} className="relative shadow-lg rounded-2xl border border-slate-200 bg-white focus-within:ring-2 focus-within:ring-blue-500/50 transition-all">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={needsClarification ? "Reply to the agent here..." : "Ask the multi-agent system to research..."}
                disabled={isLoading}
                className="w-full py-4 pl-4 pr-14 bg-transparent focus:outline-none disabled:opacity-50 text-slate-700 rounded-2xl"
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="absolute right-2 top-2 bottom-2 bg-blue-600 text-white p-2 rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:bg-slate-200 disabled:text-slate-400 transition-colors flex items-center justify-center"
              >
                <Send className="w-5 h-5" />
              </button>
            </form>
            <div className="text-center mt-3 text-xs text-slate-400">
              AlphaResearch AI can make mistakes. Consider verifying important financial data.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}