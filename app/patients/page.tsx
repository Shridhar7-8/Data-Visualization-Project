import { PatientTable } from "@/components/dashboard/patient-table"
import { RiskDistribution } from "@/components/dashboard/risk-distribution"

export default function PatientsPage() {
  return (
    <div className="flex flex-col">
      <div className="border-b">
        <div className="flex h-16 items-center px-4">
          <h1 className="text-lg font-semibold">Patient Risk Analysis</h1>
        </div>
      </div>
      <div className="flex-1 space-y-4 p-4 md:p-8 pt-6">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <RiskDistribution className="lg:col-span-1" />
          <PatientTable />
        </div>
      </div>
    </div>
  )
}
