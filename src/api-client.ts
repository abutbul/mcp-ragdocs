import { QdrantClient } from '@qdrant/js-client-rest';
import OpenAI from 'openai';
import { chromium } from 'playwright';
import { McpError, ErrorCode } from '@modelcontextprotocol/sdk/types.js';
import axios from 'axios';

// Environment variables for configuration
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL;
const QDRANT_URL = process.env.QDRANT_URL;
const QDRANT_API_KEY = process.env.QDRANT_API_KEY;

if (!QDRANT_URL) {
  throw new Error('QDRANT_URL environment variable is required for cloud storage');
}

if (!QDRANT_API_KEY) {
  throw new Error('QDRANT_API_KEY environment variable is required for cloud storage');
}

export class ApiClient {
  qdrantClient: QdrantClient;
  openaiClient?: OpenAI;
  browser: any;

  constructor() {
    // Initialize Qdrant client with cloud configuration
    this.qdrantClient = new QdrantClient({
      url: QDRANT_URL,
      apiKey: QDRANT_API_KEY,
    });

    // Initialize OpenAI client if API key is provided
    if (OPENAI_API_KEY) {
      this.openaiClient = new OpenAI({
        apiKey: OPENAI_API_KEY,
        baseURL: OPENAI_BASE_URL,
      });
    }
  }

  async initBrowser() {
    if (!this.browser) {
      this.browser = await chromium.launch();
    }
  }

  async cleanup() {
    if (this.browser) {
      await this.browser.close();
    }
  }

  async getEmbeddings(text: string): Promise<number[]> {
    try {
      // Using direct Axios call for Ollama compatibility
      if (OPENAI_BASE_URL) {
        // Make sure we're connecting to the embeddings endpoint
        const ollamaUrl = OPENAI_BASE_URL.endsWith('/v1') 
          ? `${OPENAI_BASE_URL}/embeddings` 
          : `${OPENAI_BASE_URL}/v1/embeddings`;
        
        const response = await axios.post(ollamaUrl, {
          model: 'nomic-embed-text',
          input: text,
        }, {
          headers: {
            'Content-Type': 'application/json',
            ...(OPENAI_API_KEY && { 'Authorization': `Bearer ${OPENAI_API_KEY}` })
          }
        });
        
        if (response.data && response.data.data && response.data.data[0] && response.data.data[0].embedding) {
          return response.data.data[0].embedding;
        }
        throw new Error('Invalid response format from Ollama embeddings API');
      } else {
        // Fallback to OpenAI client if OPENAI_BASE_URL is not specified
        if (!this.openaiClient) {
          throw new McpError(
            ErrorCode.InvalidRequest,
            'OpenAI API key not configured'
          );
        }

        const response = await this.openaiClient.embeddings.create({
          model: 'nomic-embed-text',
          input: text,
        });
        return response.data[0].embedding;
      }
    } catch (error) {
      console.error('Embedding error details:', error);
      
      // Generate fallback embeddings if API call fails
      console.warn('Generating fallback random embeddings');
      return this.generateRandomEmbedding(768);
    }
  }
  
  private generateRandomEmbedding(dimension: number): number[] {
    const embedding = Array.from({ length: dimension }, () => Math.random() * 2 - 1);
    const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return embedding.map(val => val / magnitude);
  }

  async initCollection(COLLECTION_NAME: string) {
    try {
      const collections = await this.qdrantClient.getCollections();
      const exists = collections.collections.some(c => c.name === COLLECTION_NAME);

      if (!exists) {
        await this.qdrantClient.createCollection(COLLECTION_NAME, {
          vectors: {
            size: 768, // nomic-embed-text:latest embedding size
            distance: 'Cosine',
          },
          // Add optimized settings for cloud deployment
          optimizers_config: {
            default_segment_number: 2,
            memmap_threshold: 20000,
          },
          replication_factor: 2,
        });
      }
    } catch (error) {
      if (error instanceof Error) {
        if (error.message.includes('unauthorized')) {
          throw new McpError(
            ErrorCode.InvalidRequest,
            'Failed to authenticate with Qdrant cloud. Please check your API key.'
          );
        } else if (error.message.includes('ECONNREFUSED') || error.message.includes('ETIMEDOUT')) {
          throw new McpError(
            ErrorCode.InternalError,
            'Failed to connect to Qdrant cloud. Please check your QDRANT_URL.'
          );
        }
      }
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to initialize Qdrant cloud collection: ${error}`
      );
    }
  }
}