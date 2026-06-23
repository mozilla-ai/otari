export interface DemoModel {
  id: string;
  label: string;
  price: number; // proxy $ per 1k calls
}

export interface DemoResponse {
  model: string;
  label: string;
  text: string;
  score: number;
}

export interface DemoItem {
  id: string;
  task: string;
  prompt: string;
  responses: DemoResponse[];
  embedding: number[];
}

export interface DemoData {
  models: DemoModel[];
  items: DemoItem[];
}
