import { type RouteConfig, index } from "@react-router/dev/routes";

export default [
  index("routes/question.tsx"),
  {
    path: "mcq",
    file: "routes/mcq.tsx",
  },
] satisfies RouteConfig;
