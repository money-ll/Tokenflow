const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000/api/v1";

export async function optimizeFile(file, query = "", targetReduction = 0.45) {
  const form = new FormData();
  form.append("file", file);
  form.append("query", query);
  form.append("target_reduction", String(targetReduction));

  const response = await fetch(`${API_BASE}/optimize`, {
    method: "POST",
    body: form,
  });

  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Optimization failed.");
  return data;
}

export async function getHistory() {
  const response = await fetch(`${API_BASE}/history`);
  return response.json();
}
