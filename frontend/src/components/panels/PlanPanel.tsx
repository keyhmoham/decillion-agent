import { Divider } from "antd";
import { useChat } from "../../store/useStore";
import { TaskBoard, TodoList } from "../chat/Blocks";

export function PlanPanel() {
  const chat = useChat();
  return (
    <div className="panel-pad">
      <h4>Agent plan</h4>
      <TodoList todos={chat.todos} />
      <Divider />
      <h4>Team board</h4>
      {chat.tasks.length ? <TaskBoard tasks={chat.tasks} /> : <div className="muted pad">No team run yet. Use <b>Team</b> mode (or ask for a team) to dispatch tasks to specialist agents working in parallel.</div>}
    </div>
  );
}
