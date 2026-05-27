import { useState, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { SquarePen, Brain, Send, StopCircle, Search, MessageSquare, Database, FileUp, X } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface InputFormProps {
  onSubmit: (inputValue: string, mode: string) => void;
  onCancel: () => void;
  isLoading: boolean;
  hasHistory: boolean;
}

export const InputForm: React.FC<InputFormProps> = ({
  onSubmit,
  onCancel,
  isLoading,
  hasHistory,
}) => {
  const [internalInputValue, setInternalInputValue] = useState("");
  const [mode, setMode] = useState("chat");
  const [pendingFile, setPendingFile] = useState<{ name: string; base64: string } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragCounterRef = useRef(0);

  const readFileAsBase64 = useCallback((file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        // Strip the data:application/pdf;base64, prefix
        const base64 = result.split(",")[1] || "";
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }, []);

  const handleFile = useCallback(async (file: File) => {
    if (file.type !== "application/pdf") {
      alert("Only PDF files are supported.");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      alert("File too large. Maximum size is 20MB.");
      return;
    }
    const base64 = await readFileAsBase64(file);
    setPendingFile({ name: file.name, base64 });
    setMode("ingest");
  }, [readFileAsBase64]);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current++;
    if (e.dataTransfer.types.includes("Files")) {
      setIsDragging(true);
    }
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current--;
    if (dragCounterRef.current === 0) {
      setIsDragging(false);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    dragCounterRef.current = 0;

    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    // Reset input so the same file can be selected again
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, [handleFile]);

  const handleInternalSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();

    if (pendingFile) {
      const prompt = internalInputValue.trim();
      const pdfTag = `[UPLOAD_PDF:${pendingFile.name}]${pendingFile.base64}[/UPLOAD_PDF]`;
      const msg = prompt ? `${prompt} ${pdfTag}` : pdfTag;
      onSubmit(msg, "ingest");
      setPendingFile(null);
      setInternalInputValue("");
      return;
    }

    if (!internalInputValue.trim()) return;
    onSubmit(internalInputValue, mode);
    setInternalInputValue("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
      e.preventDefault();
      handleInternalSubmit();
    }
  };

  const isSubmitDisabled = (!internalInputValue.trim() && !pendingFile) || isLoading;

  return (
    <form
      onSubmit={handleInternalSubmit}
      className="relative flex flex-col gap-2 p-3 pb-4"
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {/* Drag overlay */}
      {isDragging && (
        <div className="absolute inset-0 z-50 flex items-center justify-center rounded-3xl border-2 border-dashed border-blue-400 bg-blue-400/10 backdrop-blur-sm pointer-events-none">
          <div className="flex flex-col items-center gap-2 text-blue-300">
            <FileUp className="h-10 w-10" />
            <span className="text-lg font-medium">Drop PDF here</span>
          </div>
        </div>
      )}

      {/* Pending file indicator */}
      {pendingFile && (
        <div className="flex items-center gap-2 px-3 py-2 bg-neutral-600 rounded-xl text-sm text-neutral-200">
          <FileUp className="h-4 w-4 text-yellow-400 shrink-0" />
          <span className="truncate">{pendingFile.name}</span>
          <button
            type="button"
            onClick={() => setPendingFile(null)}
            className="ml-auto text-neutral-400 hover:text-neutral-200 cursor-pointer"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      <div
        className={`relative flex flex-row items-center justify-between text-white rounded-3xl rounded-bl-sm ${
          hasHistory ? "rounded-br-sm" : ""
        } break-words min-h-7 bg-neutral-700 px-4 pt-3`}
      >
        <Textarea
          value={internalInputValue}
          onChange={(e) => setInternalInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question, research a topic, or paste a URL to ingest... (Enter to send, Ctrl+Enter for new line)"
          className="w-full text-neutral-100 placeholder-neutral-500 resize-none border-0 focus:outline-none focus:ring-0 outline-none focus-visible:ring-0 shadow-none md:text-base min-h-[56px] max-h-[200px]"
          rows={1}
        />
        <div className="-mt-3">
          {isLoading ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="text-red-500 hover:text-red-400 hover:bg-red-500/10 p-2 cursor-pointer rounded-full transition-all duration-200"
              onClick={onCancel}
            >
              <StopCircle className="h-5 w-5" />
            </Button>
          ) : (
            <Button
              type="submit"
              variant="ghost"
              className={`${
                isSubmitDisabled
                  ? "text-neutral-500"
                  : "text-blue-500 hover:text-blue-400 hover:bg-blue-500/10"
              } p-2 cursor-pointer rounded-full transition-all duration-200 text-base`}
              disabled={isSubmitDisabled}
            >
              Send
              <Send className="h-5 w-5" />
            </Button>
          )}
        </div>
      </div>
      <div className="flex items-center gap-2 overflow-x-auto">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <div className="flex items-center gap-2 bg-neutral-700 border-neutral-600 text-neutral-300 rounded-xl rounded-t-sm pl-2 shrink-0">
            <div className="flex items-center text-sm whitespace-nowrap">
              <Brain className="h-4 w-4 mr-2" />
              Mode
            </div>
            <Select value={mode} onValueChange={setMode}>
              <SelectTrigger className="w-[130px] bg-transparent border-none cursor-pointer">
                <SelectValue placeholder="Mode" />
              </SelectTrigger>
              <SelectContent className="bg-neutral-700 border-neutral-600 text-neutral-300 cursor-pointer">
                <SelectItem
                  value="chat"
                  className="hover:bg-neutral-600 focus:bg-neutral-600 cursor-pointer"
                >
                  <div className="flex items-center">
                    <MessageSquare className="h-4 w-4 mr-2 text-green-400" /> Chat
                  </div>
                </SelectItem>
                <SelectItem
                  value="research"
                  className="hover:bg-neutral-600 focus:bg-neutral-600 cursor-pointer"
                >
                  <div className="flex items-center">
                    <Search className="h-4 w-4 mr-2 text-blue-400" /> Research
                  </div>
                </SelectItem>
                <SelectItem
                  value="memory_edit"
                  className="hover:bg-neutral-600 focus:bg-neutral-600 cursor-pointer"
                >
                  <div className="flex items-center">
                    <Database className="h-4 w-4 mr-2 text-purple-400" /> Memory
                  </div>
                </SelectItem>
                <SelectItem
                  value="ingest"
                  className="hover:bg-neutral-600 focus:bg-neutral-600 cursor-pointer"
                >
                  <div className="flex items-center">
                    <FileUp className="h-4 w-4 mr-2 text-yellow-400" /> Ingest
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
          <Button
            type="button"
            variant="ghost"
            className="bg-neutral-700 border-neutral-600 text-neutral-300 hover:text-neutral-100 hover:bg-neutral-600 cursor-pointer rounded-xl rounded-t-sm px-3 shrink-0"
            onClick={() => fileInputRef.current?.click()}
            title="Upload PDF"
          >
            <FileUp className="h-4 w-4" />
            <span className="text-sm ml-1">PDF</span>
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            className="hidden"
            onChange={handleFileSelect}
          />
        </div>
        {hasHistory && (
          <Button
            className="bg-neutral-700 border-neutral-600 text-neutral-300 cursor-pointer rounded-xl rounded-t-sm pl-2 shrink-0"
            variant="default"
            onClick={() => window.location.reload()}
          >
            <SquarePen size={16} />
            New Chat
          </Button>
        )}
      </div>
    </form>
  );
};
