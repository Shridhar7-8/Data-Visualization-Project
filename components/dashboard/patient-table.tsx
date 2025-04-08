"use client"

import { useState } from "react"
import { MoreHorizontal, Search } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"

interface Patient {
  id: string
  name: string
  age: number
  admissionDate: string
  dischargeDate: string
  riskScore: number
  primaryDiagnosis: string
}

const patients: Patient[] = [
  {
    id: "P-1001",
    name: "John Smith",
    age: 72,
    admissionDate: "2023-10-15",
    dischargeDate: "2023-10-22",
    riskScore: 0.82,
    primaryDiagnosis: "Heart Failure",
  },
  {
    id: "P-1002",
    name: "Maria Garcia",
    age: 65,
    admissionDate: "2023-10-18",
    dischargeDate: "2023-10-25",
    riskScore: 0.75,
    primaryDiagnosis: "COPD",
  },
  {
    id: "P-1003",
    name: "Robert Johnson",
    age: 58,
    admissionDate: "2023-10-20",
    dischargeDate: "2023-10-24",
    riskScore: 0.45,
    primaryDiagnosis: "Pneumonia",
  },
  {
    id: "P-1004",
    name: "Sarah Williams",
    age: 81,
    admissionDate: "2023-10-12",
    dischargeDate: "2023-10-23",
    riskScore: 0.88,
    primaryDiagnosis: "Stroke",
  },
  {
    id: "P-1005",
    name: "David Lee",
    age: 67,
    admissionDate: "2023-10-19",
    dischargeDate: "2023-10-26",
    riskScore: 0.62,
    primaryDiagnosis: "Diabetes",
  },
]

export function PatientTable() {
  const [searchTerm, setSearchTerm] = useState("")

  const filteredPatients = patients.filter(
    (patient) =>
      patient.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      patient.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      patient.primaryDiagnosis.toLowerCase().includes(searchTerm.toLowerCase()),
  )

  const getRiskBadge = (score: number) => {
    if (score >= 0.7) {
      return <Badge variant="destructive">High Risk</Badge>
    } else if (score >= 0.4) {
      return (
        <Badge variant="outline" className="bg-amber-100 text-amber-800 hover:bg-amber-100">
          Medium Risk
        </Badge>
      )
    } else {
      return (
        <Badge variant="outline" className="bg-emerald-100 text-emerald-800 hover:bg-emerald-100">
          Low Risk
        </Badge>
      )
    }
  }

  return (
    <Card className="col-span-4">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>High-Risk Patients</CardTitle>
            <CardDescription>Recently discharged patients with high readmission risk</CardDescription>
          </div>
          <div className="relative w-64">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search patients..."
              className="pl-8"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Patient ID</TableHead>
              <TableHead>Name</TableHead>
              <TableHead>Age</TableHead>
              <TableHead>Discharge Date</TableHead>
              <TableHead>Primary Diagnosis</TableHead>
              <TableHead>Risk Level</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredPatients.map((patient) => (
              <TableRow key={patient.id}>
                <TableCell className="font-medium">{patient.id}</TableCell>
                <TableCell>{patient.name}</TableCell>
                <TableCell>{patient.age}</TableCell>
                <TableCell>{patient.dischargeDate}</TableCell>
                <TableCell>{patient.primaryDiagnosis}</TableCell>
                <TableCell>{getRiskBadge(patient.riskScore)}</TableCell>
                <TableCell className="text-right">
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon">
                        <MoreHorizontal className="h-4 w-4" />
                        <span className="sr-only">Open menu</span>
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem>View details</DropdownMenuItem>
                      <DropdownMenuItem>Schedule follow-up</DropdownMenuItem>
                      <DropdownMenuItem>Assign care manager</DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}
