import OpenAI from "openai";
import type { Channel, DefaultGenerics, Event, StreamChat } from "stream-chat";
import type { AIAgent } from "../types";

export class OpenAIAgent implements AIAgent {
  private openai?: OpenAI;
  private lastInteractionTs = Date.now();
  private conversationHistory: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [];
  private isGenerating = false;

  constructor(
    readonly chatClient: StreamChat,
    readonly channel: Channel
  ) {}

  dispose = async () => {
    this.chatClient.off("message.new", this.handleMessage);
    await this.chatClient.disconnectUser();
  };

  get user() {
    return this.chatClient.user;
  }

  getLastInteraction = (): number => this.lastInteractionTs;

  init = async () => {
    const apiKey = process.env.GEMINI_API_KEY as string | undefined;
    if (!apiKey) {
      throw new Error("Gemini API key is required");
    } 

    this.openai = new OpenAI({ 
      apiKey, 
      baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/"
    });

    // Initialize conversation with system prompt
    this.conversationHistory = [
      {
        role: "system",
        content: this.getWritingAssistantPrompt()
      }
    ];

    this.chatClient.on("message.new", this.handleMessage);
  };

  private getWritingAssistantPrompt = (context?: string): string => {
    const currentDate = new Date().toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
    return `You are an expert AI Writing Assistant. Your primary purpose is to be a collaborative writing partner.

**Your Core Capabilities:**
- Content Creation, Improvement, Style Adaptation, Brainstorming, and Writing Coaching.
- **Current Date**: Today's date is ${currentDate}. Please use this for any time-sensitive queries.

**Response Format:**
- Be direct and production-ready.
- Use clear formatting.
- Never begin responses with phrases like "Here's the edit:", "Here are the changes:", or similar introductory statements.
- Provide responses directly and professionally without unnecessary preambles.

**Writing Context**: ${context || "General writing assistance."}

Your goal is to provide accurate, current, and helpful written content.`;
  };

  private handleMessage = async (e: Event<DefaultGenerics>) => {
    if (!this.openai || this.isGenerating) {
      console.log("OpenAI not initialized or already generating");
      return;
    }

    if (!e.message || e.message.ai_generated) {
      return;
    }

    const message = e.message.text;
    if (!message) return;

    this.lastInteractionTs = Date.now();
    this.isGenerating = true;

    try {
      // Add user message to conversation history
      this.conversationHistory.push({
        role: "user",
        content: message
      });

      // Keep conversation history manageable (last 20 messages)
      if (this.conversationHistory.length > 21) { // 1 system + 20 messages
        this.conversationHistory = [
          this.conversationHistory[0], // Keep system message
          ...this.conversationHistory.slice(-19) // Keep last 19 messages
        ];
      }

      // Send initial empty AI message
      const { message: channelMessage } = await this.channel.sendMessage({
        text: "",
        ai_generated: true,
      });

      // Send thinking indicator
      await this.channel.sendEvent({
        type: "ai_indicator.update",
        ai_state: "AI_STATE_THINKING",
        cid: channelMessage.cid,
        message_id: channelMessage.id,
      });

      // Check if user is asking for web search
      const needsWebSearch = this.shouldPerformWebSearch(message);
      let finalMessage = message; 

      if (needsWebSearch) {
        await this.channel.sendEvent({
          type: "ai_indicator.update",
          ai_state: "AI_STATE_EXTERNAL_SOURCES",
          cid: channelMessage.cid,
          message_id: channelMessage.id,
        });

        const searchResult = await this.performWebSearch(message);
        finalMessage = `User query: ${message}\n\nWeb search results: ${searchResult}\n\nPlease provide a comprehensive answer based on the search results above.`;
      }

      // Generate response using streaming
      await this.channel.sendEvent({
        type: "ai_indicator.update",
        ai_state: "AI_STATE_GENERATING",
        cid: channelMessage.cid,
        message_id: channelMessage.id,
      });

      const stream = await this.openai.chat.completions.create({
        model: "gemini-1.5-flash",
        messages: [
          ...this.conversationHistory.slice(0, -1), // All messages except the last one
          { role: "user", content: finalMessage } // Use the processed message
        ],
        stream: true,
        temperature: 0.7,
      });

      let fullResponse = "";
      let lastUpdate = Date.now();

      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta?.content;
        if (delta) {
          fullResponse += delta;
          
          // Update message every second to avoid rate limiting
          const now = Date.now();
          if (now - lastUpdate > 1000) {
            await this.chatClient.partialUpdateMessage(channelMessage.id, {
              set: { text: fullResponse },
            });
            lastUpdate = now;
          }
        }
      }

      // Final update with complete response
      await this.chatClient.partialUpdateMessage(channelMessage.id, {
        set: { text: fullResponse },
      });

      // Add AI response to conversation history
      this.conversationHistory.push({
        role: "assistant",
        content: fullResponse
      });

      // Clear AI indicator
      await this.channel.sendEvent({
        type: "ai_indicator.clear",
        cid: channelMessage.cid,
        message_id: channelMessage.id,
      });

    } catch (error) {
      console.error("Error generating response:", error);
      await this.handleError(error as Error);
    } finally {
      this.isGenerating = false;
    }
  };

  private shouldPerformWebSearch = (message: string): boolean => {
    const searchKeywords = [
      "search", "find", "lookup", "current", "recent", "news", 
      "latest", "today", "yesterday", "this week", "what's new",
      "trending", "happening", "update", "information about"
    ];
    
    const lowerMessage = message.toLowerCase();
    return searchKeywords.some(keyword => lowerMessage.includes(keyword)) ||
           lowerMessage.includes("?"); // Questions often benefit from search
  };

  private performWebSearch = async (query: string): Promise<string> => {
    const TAVILY_API_KEY = process.env.TAVILY_API_KEY;

    if (!TAVILY_API_KEY) {
      return JSON.stringify({
        error: "Web search is not available. API key not configured.",
      });
    }

    console.log(`Performing web search for: "${query}"`);

    try {
      const response = await fetch("https://api.tavily.com/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${TAVILY_API_KEY}`,
        },
        body: JSON.stringify({
          query: query,
          search_depth: "advanced",
          max_results: 5,
          include_answer: true,
          include_raw_content: false,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Tavily search failed for query "${query}":`, errorText);
        return JSON.stringify({
          error: `Search failed with status: ${response.status}`,
          details: errorText,
        });
      }

      const data = await response.json();
      console.log(`Tavily search successful for query "${query}"`);

      return JSON.stringify(data);
    } catch (error) {
      console.error(
        `An exception occurred during web search for "${query}":`,
        error
      );
      return JSON.stringify({
        error: "An exception occurred during the search.",
        message: error instanceof Error ? error.message : "Unknown error",
      });
    }
  };

  private handleError = async (error: Error) => {
    // Since we don't have a specific message reference here, 
    // we'll send a general error message
    await this.channel.sendMessage({
      text: `Error: ${error.message || "An error occurred while generating the response"}`,
      ai_generated: true,
    });
  };
}