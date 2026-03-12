import { useState } from 'react'

interface Props {
  data?: Record<string, unknown>
}

export function ResultTable({ data }: Props) {
  const [expanded, setExpanded] = useState(false)

  if (!data) return null
  const records = (data.data as Record<string, unknown>[]) || []
  if (records.length === 0) return <div className="no-data">无数据</div>

  const columns = Object.keys(records[0])
  const displayRecords = expanded ? records : records.slice(0, 10)

  const downloadJSON = () => {
    const blob = new Blob([JSON.stringify(records, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'result.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  const downloadCSV = () => {
    const header = columns.join(',')
    const rows = records.map(r =>
      columns.map(c => {
        const val = String(r[c] ?? '')
        return val.includes(',') || val.includes('"') ? `"${val.replace(/"/g, '""')}"` : val
      }).join(',')
    )
    const csv = [header, ...rows].join('\n')
    const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'result.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="result-table-wrap">
      <div className="table-scroll">
        <table className="result-table">
          <thead>
            <tr>{columns.map(c => <th key={c}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {displayRecords.map((row, i) => (
              <tr key={i}>
                {columns.map(c => (
                  <td key={c}>{String(row[c] ?? '')}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="result-actions">
        {records.length > 10 && (
          <button onClick={() => setExpanded(!expanded)}>
            {expanded ? '收起' : `展开全部 (${records.length} 条)`}
          </button>
        )}
        <button onClick={downloadJSON}>下载 JSON</button>
        <button onClick={downloadCSV}>下载 CSV</button>
      </div>
    </div>
  )
}
