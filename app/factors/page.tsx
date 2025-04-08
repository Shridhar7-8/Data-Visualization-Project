import { FactorImportance } from "@/components/dashboard/factor-importance"
import { ReadmissionTrend } from "@/components/dashboard/readmission-trend"
import { ModelPerformance } from "@/components/dashboard/model-performance"

export default function FactorsPage() {
  return (
    <div className="flex flex-col">
      <div className="border-b">
        <div className="flex h-16 items-center px-4">
          <h1 className="text-lg font-semibold">Readmission Factors Analysis</h1>
        </div>
      </div>
      <div className="flex-1 space-y-4 p-4 md:p-8 pt-6">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <FactorImportance className="lg:col-span-2" />
          <ReadmissionTrend className="lg:col-span-2" />
        </div>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <ModelPerformance />
        </div>
      </div>
    </div>
  )
}
